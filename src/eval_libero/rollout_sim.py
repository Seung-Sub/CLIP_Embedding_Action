"""LIBERO 폐루프 평가 — suite의 각 태스크를 실시간 추론으로 수행, 성공률 측정.

루프 (20Hz, receding horizon):
  8스텝마다: agentview 렌더 → (방향 보정) → CLIP 인코딩 z_t
             → f(z_{t−16}, z_t, g(A_past)) → h(ζ̂, z_t) → 16스텝 → 앞 8스텝 실행
성공 판정: env.check_success() (LIBERO 표준, 태스크별 고정 초기상태 세트 사용)

참고: 데모와 env 렌더는 동일 방향임을 실측으로 확인 (공식 코드의 [::-1]은 영상표시용).

사용 (clip_libero env):
  MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --suite libero_spatial --episodes 10
  MUJOCO_GL=egl python src/eval_libero/rollout_sim.py --task-id 0 --episodes 20 --save-video 2
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse
import collections
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image

from core import chunkrep
from core.clip_wrapper import ClipWrapper
from data.libero import LiberoDataset
from eval_libero.rollout_dataset import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=None,
                    help="특정 태스크만 (기본: suite 전체)")
    ap.add_argument("--episodes", type=int, default=10, help="태스크당 롤아웃 수")
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--flip", action=argparse.BooleanOptionalAction, default=False,
                    help="env 렌더 상하반전 (실측: 데모와 동일 방향 — 기본 off)")
    ap.add_argument("--save-video", type=int, default=0)
    ap.add_argument("--instruction-mode", choices=["correct", "wrong", "blank"],
                    default="correct",
                    help="언어 사용 판별(§0.6 R2): correct(정상)/wrong(다른 태스크 지시문)/"
                         "blank(빈 문자열). 언어를 실제 쓰면 wrong·blank에서 SR이 하락해야 정상")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam) = load_models(cfg, device)
    ds = LiberoDataset(cfg)          # span/resample 재사용
    clip = ClipWrapper()
    span, H = ds.span, args.exec_horizon

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None \
        else list(range(suite.get_num_tasks()))
    n_tasks = suite.get_num_tasks()
    if args.instruction_mode != "correct" and not use_lang:
        print(f"경고: instruction-mode={args.instruction_mode}이나 정책에 언어 토큰 없음"
              " (use_lang=False) — 판별 대조 무의미 (correct와 동일).", flush=True)

    def instruction_for(tid):
        """판별평가 지시문 선택. wrong = 순환 오프셋(n//2)으로 결정론적 불일치."""
        if args.instruction_mode == "blank":
            return ""
        if args.instruction_mode == "wrong":
            return suite.get_task((tid + n_tasks // 2) % n_tasks).language
        return suite.get_task(tid).language

    videos_dir = WS / "outputs" / "eval" / "videos"
    results = {}

    for tid in task_ids:
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=128, camera_widths=128)
        init_states = suite.get_task_init_states(tid)
        lang = torch.tensor(
            clip.encode_texts([instruction_for(tid)])["embeds"][0][None],
            device=device) if use_lang else None
        succ, infer_ms = [], []

        def frame(obs):
            img = obs["agentview_image"]
            return img[::-1].copy() if args.flip else img

        def encode(obs):
            return clip.encode_images([Image.fromarray(frame(obs))])["embeds"][0]

        def encode_wrist(obs):
            img = obs["robot0_eye_in_hand_image"]
            img = img[::-1].copy() if args.flip else img
            return clip.encode_images([Image.fromarray(img)])["embeds"][0]

        for ep in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(5):                       # 물리 안정화 (LIBERO 관례)
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            rest = np.array([0.0] * 6 + [-1.0])
            past_actions = collections.deque([rest.copy() for _ in range(span)],
                                             maxlen=span)
            z_hist = collections.deque([encode(obs)], maxlen=span // H + 1)
            frames, done, t = [], False, 0
            with torch.no_grad():
                while t < args.max_steps and not done:
                    t0 = time.time()
                    past = ds.resample_chunk(np.stack(past_actions))
                    past = ((past - a_mean) / a_std).astype(np.float32)
                    past = chunkrep.to_repr(past, repr_kind)
                    zp = torch.tensor(z_hist[0][None], device=device)
                    zc = torch.tensor(z_hist[-1][None], device=device)
                    a_emb = ae.g(torch.tensor(past[None], device=device), zp)
                    toks = [zp, zc, a_emb] + ([lang] if use_lang else []) \
                        + ([torch.tensor(encode_wrist(obs)[None], device=device)]
                           if wrist_cam else [])
                    zeta = policy(torch.stack(toks, dim=1))
                    ahat = chunkrep.from_repr(
                        ae.h(zeta, zc).cpu().numpy()[0], repr_kind) \
                        * a_std + a_mean
                    ahat = np.clip(ahat, -1.0, 1.0)
                    infer_ms.append((time.time() - t0) * 1000)
                    for k in range(min(H, args.max_steps - t)):
                        obs, r, done, info = env.step(ahat[k])
                        past_actions.append(ahat[k].copy())
                        if ep < args.save_video:
                            frames.append(frame(obs)[::-1])   # 모델 입력(frame)은 비반전 유지,
                                                              # 영상은 사람이 보기 위해 반전
                        t += 1
                        if done:
                            break
                    z_hist.append(encode(obs))
            ok = bool(done)                          # LIBERO: done == success
            succ.append(ok)
            print(f"[task {tid}] ep {ep:2d} | {'SUCCESS' if ok else 'fail'} "
                  f"| steps {t} | 추론 {np.mean(infer_ms):.1f}ms", flush=True)
            if ep < args.save_video and frames:
                import imageio
                videos_dir.mkdir(parents=True, exist_ok=True)
                vp = videos_dir / f"libero_t{tid}_ep{ep}_{'ok' if ok else 'fail'}.mp4"
                imageio.mimsave(vp, frames, fps=20)
        env.close()
        sr = float(np.mean(succ)) * 100
        results[tid] = sr
        print(f"== task {tid} [{task.language[:50]}]: {sr:.0f}% "
              f"({int(np.sum(succ))}/{args.episodes})", flush=True)

    print(f"\n=== {args.suite} | 태스크당 {args.episodes} 롤아웃 | instr={args.instruction_mode} ===")
    for tid, sr in results.items():
        print(f"task {tid:2d}: {sr:5.1f}%  {suite.get_task(tid).language[:60]}")
    print(f"평균 성공률: {np.mean(list(results.values())):.1f}%")
    out = WS / "outputs" / "eval" / f"rollout_{args.suite}_{args.instruction_mode}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"task{t}: {s:.1f}%" for t, s in results.items())
                   + f"\nmean: {np.mean(list(results.values())):.1f}%\n")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
