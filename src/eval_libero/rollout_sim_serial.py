"""LIBERO 폐루프 다단계 연쇄 평가 — 주기적 카메라 재조회 (world-model rollforward).

표준 폐루프(rollout_sim.py)는 8스텝(exec-horizon)마다 카메라를 다시 렌더·인코딩해
z_t를 갱신한다. 이 스크립트는 span(16)스텝 단위로 결정하며, 재조회 주기 사이에는
카메라를 쓰지 않고 순수 상상으로 잠재상태를 전진시킨다:

  z_next = z_cur + g(디코딩한 행동, z_cur)        (world-model 전진, rollout_dataset_serial.py와 동일 메커니즘)

행동 자체는 항상 실제로 env.step()에 실행된다(물리는 항상 진짜) — "블라인드"는
오직 *다음 계획에 쓰는 z_cur*가 카메라 재조회 대신 상상값으로 대체된다는 뜻이다.

--n : 재조회 주기(청크 단위, 카메라를 실제로 쓰는 간격)
  --n 1   재조회를 매 청크마다 (표준 8스텝 재계획보다 거칠지만 가장 촘촘한 폐루프)
  --n 4   3청크 블라인드 → 4번째 청크에서 재조회 → 반복
  --n 0   특수 케이스: 최초 1청크(t=0)만 실측, 이후 에피소드 끝까지 완전 블라인드
         (순수 open-loop dead-reckoning — 재조회가 아예 없음)

"블라인드"에는 이미지도 관절값(=과거 행동 히스토리 g() 입력)도 실측을 새로 받지
않는다: g()에 넣는 과거 행동은 폐루프 표준과 동일하게 이미 실행된 자기 자신의
행동(정책 출력)이라 원래도 "실측 GT"가 아니다 — 재조회 시점에만 카메라(agentview,
손목캠 사용 시 그것도) 를 새로 인코딩해 z_cur를 실측으로 교체한다.
실측 카메라는 블라인드 청크에서도 진단용 drift(cos(상상 z, 실측 z))를 위해
계속 인코딩하지만, 재조회 청크가 아니면 제어 경로에는 쓰지 않는다.

wrist_token 모델을 여기 쓰는 경우 블라인드 구간엔 손목캠 상상이 없으므로 마지막
실측 손목 임베딩을 그대로 유지한다(정지 근사) — 기본 config는 wrist 제외
(phase2_libero_nowrist.yaml)이므로 해당 없음.

사용 (clip_libero env):
  MUJOCO_GL=egl python src/eval_libero/rollout_sim_serial.py --n 0 --episodes 10
  MUJOCO_GL=egl python src/eval_libero/rollout_sim_serial.py --n 4 --task-id 0 --episodes 20
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
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero_nowrist.yaml"))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=None,
                    help="특정 태스크만 (기본: suite 전체)")
    ap.add_argument("--episodes", type=int, default=10, help="태스크당 롤아웃 수")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--n", type=int, default=0,
                    help="재조회 주기(청크 단위). 0 = 최초 1청크만 실측 후 끝까지 블라인드")
    ap.add_argument("--flip", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--save-video", type=int, default=0)
    args = ap.parse_args()
    assert args.n >= 0, "--n >= 0 (0 = 최초만 실측, 나머지는 재조회 주기)"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam) = load_models(cfg, device)
    ds = LiberoDataset(cfg)
    clip = ClipWrapper()
    span = ds.span

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None \
        else list(range(suite.get_num_tasks()))
    videos_dir = WS / "outputs" / "eval" / "videos"
    results, drift_stats = {}, {}

    for tid in task_ids:
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=128, camera_widths=128)
        init_states = suite.get_task_init_states(tid)
        lang = torch.tensor(clip.encode_texts([task.language])["embeds"][0][None],
                            device=device) if use_lang else None
        succ, infer_ms, drifts = [], [], []

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
            for _ in range(5):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            rest = np.array([0.0] * 6 + [-1.0])
            past_actions = collections.deque([rest.copy() for _ in range(span)],
                                             maxlen=span)
            z_now = encode(obs)
            z_prev_np, z_cur_np = z_now.copy(), z_now.copy()
            z_wrist_last = encode_wrist(obs) if wrist_cam else None
            frames, done, t, k = [], False, 0, 0
            with torch.no_grad():
                while t < args.max_steps and not done:
                    resync = (k == 0) or (args.n > 0 and k % args.n == 0)
                    z_real_now = encode(obs)              # 진단용(항상), 제어엔 재조회 시점만 사용
                    if resync:
                        z_cur_np = z_real_now.copy()
                    else:
                        drifts.append(float(
                            (z_cur_np * z_real_now).sum()
                            / (np.linalg.norm(z_cur_np) * np.linalg.norm(z_real_now) + 1e-8)))

                    t0 = time.time()
                    past = ds.resample_chunk(np.stack(past_actions))
                    past = ((past - a_mean) / a_std).astype(np.float32)
                    past = chunkrep.to_repr(past, repr_kind)
                    zp = torch.tensor(z_prev_np[None], device=device)
                    zc = torch.tensor(z_cur_np[None], device=device)
                    a_emb = ae.g(torch.tensor(past[None], device=device), zp)
                    toks = [zp, zc, a_emb] + ([lang] if use_lang else [])
                    if wrist_cam:
                        toks.append(torch.tensor(
                            (encode_wrist(obs) if resync else z_wrist_last)[None],
                            device=device))
                    zeta = policy(torch.stack(toks, dim=1))
                    ahat_t = ae.h(zeta, zc)
                    ahat = chunkrep.from_repr(ahat_t.cpu().numpy()[0], repr_kind) \
                        * a_std + a_mean
                    ahat = np.clip(ahat, -1.0, 1.0)
                    infer_ms.append((time.time() - t0) * 1000)

                    n_exec = min(span, args.max_steps - t)
                    for i in range(n_exec):
                        obs, r, done, info = env.step(ahat[i])
                        past_actions.append(ahat[i].copy())
                        if ep < args.save_video:
                            frames.append(frame(obs)[::-1])
                        t += 1
                        if done:
                            break

                    if wrist_cam and resync:
                        z_wrist_last = encode_wrist(obs)
                    if done:
                        break
                    delta_hat = ae.g(ahat_t, zc)                 # 디코딩한 행동을 다시 인코딩 → 신뢰할 만한 Δ
                    z_imagined = (zc + delta_hat).cpu().numpy()[0]
                    z_prev_np = z_cur_np
                    z_cur_np = z_imagined
                    k += 1
            ok = bool(done)
            succ.append(ok)
            print(f"[task {tid}] ep {ep:2d} | {'SUCCESS' if ok else 'fail'} "
                  f"| steps {t} | 청크 {k+1} | 추론 {np.mean(infer_ms):.1f}ms", flush=True)
            if ep < args.save_video and frames:
                import imageio
                videos_dir.mkdir(parents=True, exist_ok=True)
                vp = videos_dir / f"libero_serial_t{tid}_ep{ep}_n{args.n}_{'ok' if ok else 'fail'}.mp4"
                imageio.mimsave(vp, frames, fps=20)
        env.close()
        sr = float(np.mean(succ)) * 100
        results[tid] = sr
        drift_stats[tid] = float(np.mean(drifts)) if drifts else float("nan")
        print(f"== task {tid} [{task.language[:50]}]: {sr:.0f}% "
              f"({int(np.sum(succ))}/{args.episodes}) | 평균 블라인드 drift {drift_stats[tid]:.3f}",
              flush=True)

    print(f"\n=== {args.suite} | 태스크당 {args.episodes} 롤아웃 | 재조회 주기 n={args.n} ===")
    for tid, sr in results.items():
        print(f"task {tid:2d}: {sr:5.1f}%  drift {drift_stats[tid]:.3f}  "
              f"{suite.get_task(tid).language[:60]}")
    mean_sr = np.mean(list(results.values()))
    mean_drift = np.nanmean(list(drift_stats.values()))
    print(f"평균 성공률: {mean_sr:.1f}%  |  평균 블라인드 drift: {mean_drift:.3f}")
    out = WS / "outputs" / "eval" / f"rollout_{args.suite}_serial_n{args.n}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"n={args.n}\n" +
                  "\n".join(f"task{t}: {s:.1f}%  drift {drift_stats[t]:.3f}"
                            for t, s in results.items())
                  + f"\nmean: {mean_sr:.1f}%\nmean_drift: {mean_drift:.3f}\n")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
