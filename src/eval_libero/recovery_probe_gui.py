"""GUI: failure-recovery observation + language-paraphrase probe (LIBERO closed loop).

Motivation: libero_spatial has only 10 distinct language instructions, so a
policy could be memorizing those 10 exact strings rather than genuinely
grounding language -> action. This tool lets you pick a task, pick either the
original instruction or a suggested paraphrase, and run the closed loop
continuously (regardless of success/fail) so you can watch by eye whether
recovery behavior appears after a failed grasp, and whether behavior holds up
under paraphrased instructions.

Two live plots, updated every control step / replanning chunk:
  - executed 7D action trajectory for the current episode (pos/rot/gripper)
  - image-action alignment cos(g(A_chunk, z_prev), z_cur - z_prev) per chunk
    (same metric as the gripper-alignment / drift diagnostics used elsewhere
    in this project) -- a live "surprise" signal: low cos means the actual
    visual outcome diverged from what the executed actions would predict.

Start runs exactly one episode, then stops automatically (success or fail) --
press Start again for the next one. By default there is no step cap (episodes
only end on task success or a manual Stop), since the whole point is to watch
whether a recovery attempt eventually shows up after a failed grasp, which
may take longer than a fixed-length benchmark rollout would allow. Pass
--max-steps N to cap episode length if you want the old fixed-horizon
behavior back.

usage (clip_libero env, desktop session):
  MUJOCO_GL=egl python src/eval_libero/recovery_probe_gui.py
  MUJOCO_GL=egl python src/eval_libero/recovery_probe_gui.py --config configs/phase2_libero.yaml
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse
import collections
import os
import threading

import matplotlib
import matplotlib.animation
import numpy as np
import torch
import yaml
from PIL import Image

from core import chunkrep
from core.clip_wrapper import ClipWrapper
from data.libero import LiberoDataset
from eval_libero.paraphrases import PARAPHRASES
from eval_libero.rollout_dataset import load_models  # NOTE: sets a CJK font.family as a
                                                       # side effect of import - override below

FS = 2.2
matplotlib.rcParams.update({"font.family": ["sans-serif"], "font.size": 9 * FS,
                            "axes.unicode_minus": False})
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

DIM_GROUPS = [("pos (dx,dy,dz)", [0, 1, 2]), ("rot (droll,dpitch,dyaw)", [3, 4, 5]),
              ("gripper", [6])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="episode step cap. 0 = unlimited (only Stop or task success ends it)")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--selftest", type=float, default=0,
                    help="headless check: run for N seconds via Start/Stop then exit (no window)")
    args = ap.parse_args()
    if args.selftest:
        matplotlib.use("Agg")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam, *_) = load_models(cfg, device)   # *_ = F3 obs (미사용)
    ds = LiberoDataset(cfg)
    clip = ClipWrapper()
    span, H = ds.span, args.exec_horizon

    suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = suite.get_num_tasks()
    names = [suite.get_task(i).language for i in range(n_tasks)]

    # ---- shared state between GUI thread and worker thread ----
    state = {"task": args.task_id, "instr_idx": 0}
    stop_event = threading.Event()
    running = {"flag": False}
    ep_counters = collections.defaultdict(int)
    status = {"episode": 0, "success": 0, "fail": 0, "last": "-", "instr": "", "task": 0}
    frame_holder = {"frame": np.zeros((128, 128, 3), dtype=np.uint8)}

    def new_ep_buf():
        return {"t": [], "traj": {d: [] for d in range(7)}, "cos_x": [], "cos": []}

    buf_holder = {"cur": new_ep_buf()}  # swapped atomically per episode (thread-safety, see below)
    env_holder = {"env": None, "tid": None}

    def instruction_text(tid, idx):
        if idx == 0:
            return names[tid]
        return PARAPHRASES[tid][idx - 1]

    def ensure_env(tid):
        if env_holder["tid"] == tid:
            return env_holder["env"]
        if env_holder["env"] is not None:
            env_holder["env"].close()
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env_holder["env"] = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env_holder["tid"] = tid
        return env_holder["env"]

    def run_episode(tid, instr_text):
        env = ensure_env(tid)
        init_states = suite.get_task_init_states(tid)
        env.reset()
        obs = env.set_init_state(init_states[ep_counters[tid] % len(init_states)])
        ep_counters[tid] += 1
        for _ in range(5):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        rest = np.array([0.0] * 6 + [-1.0])
        past_actions = collections.deque([rest.copy() for _ in range(span)], maxlen=span)

        def frame(o):
            return o["agentview_image"]  # model input - keep unflipped (matches training)

        def display_frame(o):
            return frame(o)[::-1, ::-1]  # human viewing only - raw buffer is rotated 180

        def encode(o):
            return clip.encode_images([Image.fromarray(frame(o))])["embeds"][0]

        def encode_wrist(o):
            return clip.encode_images([Image.fromarray(o["robot0_eye_in_hand_image"])])["embeds"][0]

        z_hist = collections.deque([encode(obs)], maxlen=span // H + 1)
        lang = torch.tensor(clip.encode_texts([instr_text])["embeds"][0][None],
                            device=device) if use_lang else None
        frame_holder["frame"] = display_frame(obs)
        buf = new_ep_buf()
        buf_holder["cur"] = buf  # atomic pointer swap - redraw() never sees a half-cleared buffer

        unlimited = args.max_steps <= 0
        done, t, chunk = False, 0, 0
        with torch.no_grad():
            while (unlimited or t < args.max_steps) and not done and not stop_event.is_set():
                past = ds.resample_chunk(np.stack(past_actions))
                past = ((past - a_mean) / a_std).astype(np.float32)
                past = chunkrep.to_repr(past, repr_kind)
                zp = torch.tensor(z_hist[0][None], device=device)
                zc = torch.tensor(z_hist[-1][None], device=device)
                a_emb = ae.g(torch.tensor(past[None], device=device), zp)

                if chunk >= span // H:  # z_hist warmed up -> real Δz available
                    dz = (zc - zp)[0]
                    num = float((a_emb[0] * dz).sum())
                    den = float(a_emb[0].norm() * dz.norm() + 1e-8)
                    buf["cos_x"].append(chunk)
                    buf["cos"].append(num / den)

                toks = [zp, zc, a_emb] + ([lang] if use_lang else []) \
                    + ([torch.tensor(encode_wrist(obs)[None], device=device)] if wrist_cam else [])
                zeta = policy(torch.stack(toks, dim=1))
                ahat = chunkrep.from_repr(ae.h(zeta, zc).cpu().numpy()[0], repr_kind) * a_std + a_mean
                ahat = np.clip(ahat, -1.0, 1.0)

                for k in range(H if unlimited else min(H, args.max_steps - t)):
                    if stop_event.is_set():
                        break
                    try:
                        obs, r, done, info = env.step(ahat[k])
                    except ValueError:
                        # LIBERO's step() overwrites robosuite's horizon-based `done` with
                        # just _check_success(), so reaching the internal horizon (1000
                        # steps) never surfaces as done=True - the *next* step() call then
                        # raises "executing action in terminated episode". Treat that as
                        # the episode timing out (fail) instead of crashing.
                        done = True
                        break
                    past_actions.append(ahat[k].copy())
                    frame_holder["frame"] = display_frame(obs)
                    buf["t"].append(t)
                    for d in range(7):
                        buf["traj"][d].append(ahat[k][d])
                    t += 1
                    if done:
                        break
                z_hist.append(encode(obs))
                chunk += 1
        ok = bool(done)
        status["episode"] += 1
        status["success" if ok else "fail"] += 1
        status["last"] = "SUCCESS" if ok else "fail"
        print(f"[task {tid}] ep {status['episode']:3d} | {'SUCCESS' if ok else 'fail'} "
              f"| steps {t} | \"{instr_text[:60]}\"", flush=True)
        return ok

    def worker():
        # single episode per Start press - stops automatically when it ends
        # (success, failure, or Stop), waiting for the next Start
        tid, idx = state["task"], state["instr_idx"]
        instr = instruction_text(tid, idx)
        status["task"], status["instr"] = tid, instr
        run_episode(tid, instr)
        running["flag"] = False

    # ---------------- GUI ----------------
    fig = plt.figure(figsize=(22, 13))
    fig.canvas.manager.set_window_title("LIBERO Recovery / Paraphrase Probe") if fig.canvas.manager else None

    ax_img = fig.add_axes([0.04, 0.55, 0.30, 0.40])
    ax_img.set_title("agentview (live)", fontsize=9 * FS)
    ax_img.axis("off")
    im = ax_img.imshow(frame_holder["frame"])

    ax_traj = [fig.add_axes([0.38, 0.86 - 0.155 * i, 0.36, 0.09]) for i in range(3)]
    traj_lines = []
    for ax, (title, dims) in zip(ax_traj, DIM_GROUPS):
        ax.set_title(title, fontsize=6.5 * FS)
        ax.tick_params(labelsize=5.5 * FS)
        ax.grid(color="#EEEEEE", lw=0.5)
        lines = [ax.plot([], [], lw=1.3, label=f"d{d}")[0] for d in dims]
        traj_lines.append(lines)
    ax_traj[-1].set_xlabel("step", fontsize=6 * FS)

    ax_cos = fig.add_axes([0.04, 0.08, 0.70, 0.34])
    ax_cos.set_title("image-action alignment  cos(g(A_chunk,z_prev), z_cur-z_prev)", fontsize=8.5 * FS)
    ax_cos.set_xlabel("chunk index"); ax_cos.set_ylabel("cos")
    ax_cos.axhline(0, color="#BBBBBB", lw=1, ls=":")
    ax_cos.grid(color="#EEEEEE", lw=0.5)
    cos_line, = ax_cos.plot([], [], color="#EE6677", lw=1.6, marker="o", ms=3)

    fig.text(0.76, 0.955, "Task (LIBERO-Spatial)", fontsize=9 * FS, weight="bold")
    ax_radio_task = fig.add_axes([0.76, 0.55, 0.22, 0.38], frameon=False)
    pre = os.path.commonprefix(names)
    task_labels = [f"{i}: {n[len(pre):][:34] or n[:34]}" for i, n in enumerate(names)]
    radio_task = RadioButtons(ax_radio_task, task_labels, active=args.task_id)
    for lb in radio_task.labels:
        lb.set_fontsize(6.3 * FS)

    fig.text(0.76, 0.50, "Instruction variant", fontsize=9 * FS, weight="bold")
    ax_radio_instr = fig.add_axes([0.76, 0.28, 0.22, 0.20], frameon=False)
    instr_labels = ["0: original", "1: paraphrase A", "2: paraphrase B", "3: paraphrase C"]
    radio_instr = RadioButtons(ax_radio_instr, instr_labels, active=0)
    for lb in radio_instr.labels:
        lb.set_fontsize(7 * FS)

    txt_instr = fig.text(0.76, 0.24, "", fontsize=6.5 * FS, color="#333333", wrap=True)
    txt_status = fig.text(0.76, 0.03, "", fontsize=7.5 * FS, color="#111111")

    ax_start = fig.add_axes([0.76, 0.14, 0.10, 0.05])
    ax_stop = fig.add_axes([0.88, 0.14, 0.10, 0.05])
    btn_start = Button(ax_start, "Start", color="#DDF3DD", hovercolor="#BFEFBF")
    btn_stop = Button(ax_stop, "Stop", color="#F3DDDD", hovercolor="#EFBFBF")

    def on_task(label):
        state["task"] = int(label.split(":")[0])

    def on_instr(label):
        state["instr_idx"] = int(label.split(":")[0])

    def on_start(event):
        if running["flag"]:
            return
        stop_event.clear()
        running["flag"] = True
        threading.Thread(target=worker, daemon=True).start()

    def on_stop(event):
        stop_event.set()

    radio_task.on_clicked(on_task)
    radio_instr.on_clicked(on_instr)
    btn_start.on_clicked(on_start)
    btn_stop.on_clicked(on_stop)

    def redraw(_frame):
        im.set_data(frame_holder["frame"])
        buf = buf_holder["cur"]  # single atomic read - never a half-swapped episode buffer
        n_t = min(len(buf["t"]), min((len(buf["traj"][d]) for d in range(7)), default=0))
        for lines, (_, dims) in zip(traj_lines, DIM_GROUPS):
            for line, d in zip(lines, dims):
                line.set_data(buf["t"][:n_t], buf["traj"][d][:n_t])
        for ax, (_, dims) in zip(ax_traj, DIM_GROUPS):
            ax.relim(); ax.autoscale_view()
        n_c = min(len(buf["cos_x"]), len(buf["cos"]))
        cos_line.set_data(buf["cos_x"][:n_c], buf["cos"][:n_c])
        ax_cos.relim(); ax_cos.autoscale_view()

        idx = state["instr_idx"]
        preview = instruction_text(state["task"], idx)
        txt_instr.set_text(f'"{preview}"')
        state_str = "RUNNING episode..." if running["flag"] else "ready (press Start)"
        txt_status.set_text(
            f"[{state_str}]  task {status['task']}  episodes run: {status['episode']}\n"
            f"success {status['success']}  fail {status['fail']}  last: {status['last']}")
        btn_start.color = "#AAAAAA" if running["flag"] else "#DDF3DD"
        btn_stop.color = "#F3DDDD" if running["flag"] else "#AAAAAA"
        return []

    ani = matplotlib.animation.FuncAnimation(fig, redraw, interval=200, cache_frame_data=False)

    if args.selftest:
        import time
        on_start(None)
        t0 = time.time()
        while time.time() - t0 < args.selftest:
            redraw(0)
            time.sleep(0.2)
        on_stop(None)
        time.sleep(0.5)
        buf = buf_holder["cur"]
        print(f"selftest OK: episodes={status['episode']} success={status['success']} "
              f"fail={status['fail']} traj_pts={len(buf['t'])} cos_pts={len(buf['cos'])}")
        return

    plt.show()
    stop_event.set()


if __name__ == "__main__":
    main()
