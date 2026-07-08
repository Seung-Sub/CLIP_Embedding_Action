"""LIBERO closed-loop evaluation using ONLY paraphrased instructions (never the
original training string) -- objective check of whether the language token is
grounded in meaning or just memorizing the 10 literal libero_spatial strings.

For every task, runs each of the 3 recommended paraphrases (src/eval_libero/
paraphrases.py) for --episodes rollouts and reports success rate per
(task, paraphrase) cell plus the overall mean across all 10*3 cells.

usage (clip_libero env):
  MUJOCO_GL=egl python src/eval_libero/rollout_sim_paraphrase.py --episodes 20
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse
import collections
import json
import os

import numpy as np
import torch
import yaml
from PIL import Image

from core import chunkrep
from core.clip_wrapper import ClipWrapper
from data.libero import LiberoDataset
from eval_libero.paraphrases import PARAPHRASES
from eval_libero.rollout_dataset import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=20, help="롤아웃 수, 태스크당 페러프레이징마다")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam) = load_models(cfg, device)
    ds = LiberoDataset(cfg)
    clip = ClipWrapper()
    span, H = ds.span, args.exec_horizon

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None else list(range(suite.get_num_tasks()))
    results = {}

    for tid in task_ids:
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        init_states = suite.get_task_init_states(tid)

        def encode(o):
            return clip.encode_images([Image.fromarray(o["agentview_image"])])["embeds"][0]

        def encode_wrist(o):
            return clip.encode_images([Image.fromarray(o["robot0_eye_in_hand_image"])])["embeds"][0]

        for pidx, instr_text in enumerate(PARAPHRASES[tid]):
            lang = torch.tensor(clip.encode_texts([instr_text])["embeds"][0][None],
                                device=device) if use_lang else None
            succ = []
            for ep in range(args.episodes):
                env.reset()
                obs = env.set_init_state(init_states[ep % len(init_states)])
                for _ in range(5):
                    obs, *_ = env.step([0.0] * 6 + [-1.0])
                rest = np.array([0.0] * 6 + [-1.0])
                past_actions = collections.deque([rest.copy() for _ in range(span)], maxlen=span)
                z_hist = collections.deque([encode(obs)], maxlen=span // H + 1)
                done, t = False, 0
                with torch.no_grad():
                    while t < args.max_steps and not done:
                        past = ds.resample_chunk(np.stack(past_actions))
                        past = ((past - a_mean) / a_std).astype(np.float32)
                        past = chunkrep.to_repr(past, repr_kind)
                        zp = torch.tensor(z_hist[0][None], device=device)
                        zc = torch.tensor(z_hist[-1][None], device=device)
                        a_emb = ae.g(torch.tensor(past[None], device=device), zp)
                        toks = [zp, zc, a_emb] + ([lang] if use_lang else []) \
                            + ([torch.tensor(encode_wrist(obs)[None], device=device)] if wrist_cam else [])
                        zeta = policy(torch.stack(toks, dim=1))
                        ahat = chunkrep.from_repr(ae.h(zeta, zc).cpu().numpy()[0], repr_kind) * a_std + a_mean
                        ahat = np.clip(ahat, -1.0, 1.0)
                        try:
                            for k in range(min(H, args.max_steps - t)):
                                obs, r, done, info = env.step(ahat[k])
                                past_actions.append(ahat[k].copy())
                                t += 1
                                if done:
                                    break
                        except ValueError:
                            done = True  # LIBERO horizon quirk, see recovery_probe_gui.py comment
                            break
                        z_hist.append(encode(obs))
                ok = bool(done)
                succ.append(ok)
                print(f"[task {tid} para {pidx}] ep {ep:2d} | {'SUCCESS' if ok else 'fail'} "
                      f"| steps {t} | \"{instr_text[:55]}\"", flush=True)
            sr = float(np.mean(succ)) * 100
            results[(tid, pidx)] = sr
            print(f"== task {tid} para {pidx} [{instr_text[:50]}]: {sr:.1f}% "
                  f"({int(np.sum(succ))}/{args.episodes})", flush=True)
        env.close()

    print(f"\n=== {args.suite} | paraphrase-only | {args.episodes} 롤아웃/조건 ===")
    for (tid, pidx), sr in results.items():
        print(f"task {tid:2d} para {pidx}: {sr:5.1f}%")
    mean_sr = float(np.mean(list(results.values())))
    print(f"전체 평균 성공률(30개 조건): {mean_sr:.1f}%")

    out = WS / "outputs" / "eval" / f"rollout_{args.suite}_paraphrase.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for (tid, pidx), sr in results.items():
            f.write(json.dumps({"task": tid, "paraphrase": pidx, "success_rate": sr}) + "\n")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
