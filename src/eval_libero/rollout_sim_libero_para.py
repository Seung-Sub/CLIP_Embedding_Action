"""Experiment 1b — LIBERO-Para closed-loop eval on the libero_goal policy.

Object-lexical vs action-lexical paraphrase robustness. Mirrors
rollout_sim_paraphrase.py exactly (real libero_goal scene / init states /
success check are kept) and ONLY swaps the CLIP-encoded instruction text: for
each task it draws instruction strings from LIBERO_PARA[task_id][--para-axis].

Design: runs --episodes rollouts per task; episode ep uses paraphrase string
ep % len(strings) (cycling the frozen set) and init_state ep % len(init_states)
(same init-state schedule as the baseline correct-SR run), so SR is comparable
to the goal baseline. Writes SR per task for the chosen axis.

Pre-registered metric (owner decides): baseline = goal correct SR;
report object-lexical SR drop vs action-lexical SR drop.

usage (clip_libero env, remote GPU):
  MUJOCO_GL=egl python src/eval_libero/rollout_sim_libero_para.py \
      --config configs/phase2_libero_goal.yaml --suite libero_goal \
      --para-axis object --episodes 20
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
from eval_libero.libero_para import BASE, LIBERO_PARA, LP_EVAL
from eval_libero.rollout_dataset import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero_goal.yaml"))
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--para-axis", choices=["object", "action"], required=True,
                    help="LIBERO-Para lexical axis: object (obj noun-phrase) / action (verb)")
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=20, help="롤아웃 수 (태스크당)")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    assert args.suite == "libero_goal", \
        "LIBERO-Para strings here are frozen for the libero_goal suite only."

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam, *_) = load_models(cfg, device)   # *_ = F3 obs (미사용)
    assert use_lang, "policy has no language token — 1b (text-swap) is meaningless."
    ds = LiberoDataset(cfg)
    clip = ClipWrapper()
    span, H = ds.span, args.exec_horizon

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None else list(range(suite.get_num_tasks()))
    axis = args.para_axis
    results = {}

    for tid in task_ids:
        task = suite.get_task(tid)
        # guard: frozen strings are keyed by our task_id via base-string match
        assert task.language.strip() == BASE[tid], \
            f"alignment drift task {tid}: {task.language!r} != {BASE[tid]!r}"
        strings = LIBERO_PARA[tid][axis]
        assert strings, f"no {axis} paraphrases for task {tid}"
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        init_states = suite.get_task_init_states(tid)

        def encode(o):
            return clip.encode_images([Image.fromarray(o["agentview_image"])])["embeds"][0]

        def encode_wrist(o):
            return clip.encode_images([Image.fromarray(o["robot0_eye_in_hand_image"])])["embeds"][0]

        # pre-encode the paraphrase texts once (cheap; text swap is the whole point)
        lang_cache = [torch.tensor(clip.encode_texts([s])["embeds"][0][None], device=device)
                      for s in strings]

        succ, used = [], []
        for ep in range(args.episodes):
            instr_text = strings[ep % len(strings)]
            lang = lang_cache[ep % len(strings)]
            used.append(instr_text)
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
            print(f"[task {tid} {axis}] ep {ep:2d} | {'SUCCESS' if ok else 'fail'} "
                  f"| steps {t} | \"{instr_text[:55]}\"", flush=True)
        sr = float(np.mean(succ)) * 100
        results[tid] = {"success_rate": sr, "n_success": int(np.sum(succ)),
                        "episodes": args.episodes, "n_paraphrases": len(strings),
                        "lp_eval": LP_EVAL[tid], "base": BASE[tid]}
        print(f"== task {tid} {axis} [{BASE[tid][:45]}]: {sr:.1f}% "
              f"({int(np.sum(succ))}/{args.episodes}, {len(strings)} paraphrases)", flush=True)
        env.close()

    print(f"\n=== {args.suite} | LIBERO-Para {axis}-lexical | {args.episodes} 롤아웃/태스크 ===")
    for tid, r in results.items():
        print(f"task {tid:2d}: {r['success_rate']:5.1f}%  (eval{r['lp_eval']}) {r['base'][:50]}")
    mean_sr = float(np.mean([r["success_rate"] for r in results.values()]))
    print(f"평균 성공률: {mean_sr:.1f}%")

    tag = f"t{args.task_id}" if args.task_id is not None else "all"
    out = WS / "outputs" / "eval" / f"rollout_{args.suite}_libero_para_{axis}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"suite": args.suite, "para_axis": axis, "episodes": args.episodes,
               "mean_success_rate": mean_sr, "per_task": results,
               "source": "cau-hai-lab/LIBERO-Para (MIT), metrics/libero_para_metadata.csv"},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
