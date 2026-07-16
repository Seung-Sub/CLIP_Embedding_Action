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
from core.anchor import get_anchor
from data.libero import LiberoDataset
from eval_libero.rollout_dataset import load_models, sample_zeta


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
    ap.add_argument("--instruction-mode",
                    choices=["correct", "wrong", "blank", "swap", "v1", "v4"],
                    default="correct",
                    help="언어 사용 판별(§0.6 R2): correct(정상)/wrong(다른 태스크 지시문)/"
                         "blank(빈 문자열)/swap(1c: 같은 씬 내 타깃-스왑 — bowl_2를 가리키는 "
                         "형제 태스크 지시문). swap에서는 instructed/orig/neither 3율을 보고")
    ap.add_argument("--checkpoint", default=None,
                    help="cfg train.checkpoint 덮어쓰기 (시드/변형 롤아웃 — 별도 config 불요)")
    ap.add_argument("--ablate-zf", action="store_true", default=False,
                    help="병목-효능 프로브(C1 게이트): f4 로드 시 fine 채널 ζ_f 기여를 0으로 "
                         "만들어 pooled(ζ_g)만으로 롤아웃. 미지정 시 현행과 비트 동형. "
                         "SR(full C1) vs SR(--ablate-zf) 로 ζ_f 기여 확인.")
    ap.add_argument("--flow-fixed-noise", action="store_true", default=False,
                    help="h_mode=flow 전용: 에피소드당 x0 노이즈를 고정 시드로 → 재계획(receding-"
                         "horizon) 간 일관된 단일 모드 유지(naive fresh-noise의 mode-switching 배회 방지). "
                         "미지정 시 현행(재계획마다 독립 샘플)과 동일.")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load(open(args.config))
    if args.checkpoint:                              # 시드/변형 체크포인트 오버라이드
        cfg["train"]["checkpoint"] = args.checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam, obs_anchors, obs_fusion, f4) = load_models(cfg, device)
    is_hflow = getattr(ae, "h_mode", "mlp") == "flow"   # h가 flow 디코더면 generator 전달 가능
    ds = LiberoDataset(cfg)          # span/resample 재사용
    clip = get_anchor(cfg)          # 앵커 config 반영 (무-anchor면 ClipAnchor=ClipWrapper와 동일)
    # S1b 역할분리(cond_anchor): 조건 토큰=SigLIP2 서브블록[0:cond_dim] / g·h·ζ=융합 z 전체.
    # fused=dualconcat 이면 _sig.dim 재사용(추가 로드 없음). 없으면 cond_dim=None → 기존 비트 동형.
    cond_dim = None
    if cfg.get("cond_anchor"):
        cond_dim = (clip._sig.dim if hasattr(clip, "_sig")
                    else get_anchor({"anchor": cfg["cond_anchor"]}).dim)
    span, H = ds.span, args.exec_horizon

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None \
        else list(range(suite.get_num_tasks()))
    n_tasks = suite.get_num_tasks()
    if args.instruction_mode != "correct" and not use_lang:
        print(f"경고: instruction-mode={args.instruction_mode}이나 정책에 언어 토큰 없음"
              " (use_lang=False) — 판별 대조 무의미 (correct와 동일).", flush=True)

    # 1c 씬-내 타깃-스왑: tid → 이 씬의 bowl_2 위치를 서술하는 형제 태스크 tid.
    # (libero_spatial: 각 씬에 bowl_1=태스크목표, bowl_2=distractor)
    SWAP = {0: 1, 1: 6, 2: 8, 3: 9, 4: 9, 5: 3, 6: 7, 7: 9, 8: 1, 9: 7}

    def instruction_for(tid):
        """판별평가 지시문 선택. wrong = 순환 오프셋(n//2)으로 결정론적 불일치.
        swap = 같은 씬 bowl_2를 가리키는 형제 태스크 지시문(SWAP)."""
        if args.instruction_mode == "blank":
            return ""
        if args.instruction_mode == "wrong":
            return suite.get_task((tid + n_tasks // 2) % n_tasks).language
        if args.instruction_mode == "swap":
            return suite.get_task(SWAP[tid]).language
        # ICBench식 문자열 교란(씬/코드 불변, 지시문만): 성공기준은 원 태스크(done) 그대로 →
        # LGS = SR(correct) − SR(교란). 언어 무시 정책이면 SR 유지, 사용 정책이면 하락.
        base = suite.get_task(tid).language
        if args.instruction_mode == "v1":        # 속성 치환: black→white (씬에 white bowl 없음 = 모순)
            return base.replace("black", "white")
        if args.instruction_mode == "v4":        # 관계 치환: on→under (달성 불가 관계)
            return base.replace(" on ", " under ")
        return base

    videos_dir = WS / "outputs" / "eval" / "videos"
    results = {}
    is_swap = args.instruction_mode == "swap"
    swap_results = {}                                # tid → (instr_sr, orig_sr, neither_sr)

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
        swap_instr, swap_orig = [], []               # 1c: 지시타깃/원래타깃 도달 여부

        def frame(obs):
            img = obs["agentview_image"]
            return img[::-1].copy() if args.flip else img

        def encode(obs):
            return clip.encode_images([Image.fromarray(frame(obs))])["embeds"][0]

        def encode_wrist(obs):
            img = obs["robot0_eye_in_hand_image"]
            img = img[::-1].copy() if args.flip else img
            return clip.encode_images([Image.fromarray(img)])["embeds"][0]

        def obs_toks(obs):
            """F3: 현재 프레임의 dense patch 토큰 → obs_fusion → K개 관측 토큰.
            앵커 카메라(기본 agentview_rgb)에 맞춰 프레임 선택; zc 와 동일 프레임."""
            feat = {}
            for name, anc, cam in obs_anchors:
                if wrist_cam and cam == wrist_cam:
                    im = obs["robot0_eye_in_hand_image"]
                    im = im[::-1].copy() if args.flip else im
                else:
                    im = frame(obs)
                tok = anc.encode_images([Image.fromarray(im)])["tokens"]  # (1,P,d)
                feat[name] = torch.tensor(tok, device=device)
            ot = obs_fusion(feat)                            # (1,K,768)
            return [ot[:, k] for k in range(ot.size(1))]

        for ep in range(args.episodes):
            ep_gen = None                            # h-flow: 에피소드당 고정 노이즈(옵션)
            if args.flow_fixed_noise and is_hflow:
                ep_gen = torch.Generator(device=device)
                ep_gen.manual_seed(10000 * tid + ep)
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(5):                       # 물리 안정화 (LIBERO 관례)
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            rest = np.array([0.0] * 6 + [-1.0])
            past_actions = collections.deque([rest.copy() for _ in range(span)],
                                             maxlen=span)
            z_hist = collections.deque([encode(obs)], maxlen=span // H + 1)
            frames, done, instructed, t = [], False, False, 0
            with torch.no_grad():
                while t < args.max_steps and not done and not instructed:
                    t0 = time.time()
                    past = ds.resample_chunk(np.stack(past_actions))
                    past = ((past - a_mean) / a_std).astype(np.float32)
                    past = chunkrep.to_repr(past, repr_kind)
                    zp = torch.tensor(z_hist[0][None], device=device)
                    zc = torch.tensor(z_hist[-1][None], device=device)
                    a_emb = ae.g(torch.tensor(past[None], device=device), zp)  # g=융합 z 전체
                    _zd = zp.shape[-1]                # 융합 폭 (S1b=2048); 슬라이스 전 확정
                    # S1b: 조건 토큰(z_prev/z_cur/wrist)만 SigLIP2 서브블록[0:cond_dim]; aemb·h 는 융합 전체
                    zp_c, zc_c = (zp[:, :cond_dim], zc[:, :cond_dim]) if cond_dim else (zp, zc)
                    wr_t = torch.tensor(encode_wrist(obs)[None], device=device) if wrist_cam else None
                    if wrist_cam and cond_dim:
                        wr_t = wr_t[:, :cond_dim]
                    toks = [zp_c, zc_c, a_emb] + ([lang] if use_lang else []) \
                        + ([wr_t] if wrist_cam else [])
                    if obs_fusion is not None:       # F3: 관측 토큰 K개를 열 끝에 추가
                        toks = toks + obs_toks(obs)
                    # ζ_g(정책) + ζ_f(f4, 있으면) 를 공유-τ 단일 루프로 샘플.
                    # ζ_f 는 base 조건 noise-flow 로 생성(미래/patch ΔF 무접근).
                    # concat/S1b: 좁은 조건 토큰(lang/wrist/SigLIP2 z)→z 폭 SigLIP2 서브블록 zero-pad (기존=no-op)
                    toks = [t if t.shape[-1] == _zd else
                            torch.nn.functional.pad(t, (0, _zd - t.shape[-1])) for t in toks]
                    zeta, zeta_f = sample_zeta(policy, f4, torch.stack(toks, dim=1))
                    if args.ablate_zf and zeta_f is not None:   # 병목-효능 프로브: ζ_f 기여 0
                        zeta_f = torch.zeros_like(zeta_f)       # (미지정 시 이 분기 미실행=비트 동형)
                    ahat_lat = ae.h(zeta, zc, generator=ep_gen) if is_hflow \
                        else ae.h(zeta, zc)              # frozen h(ζ_g, z_cur); flow면 에피소드 고정노이즈
                    if f4 is not None:                   # C1: + tanh(β)·fine_head([ζ_g,ζ_f,z_cur])
                        ahat_lat = ahat_lat + f4.fine_action(zeta, zeta_f, zc)
                    ahat = chunkrep.from_repr(
                        ahat_lat.cpu().numpy()[0], repr_kind) \
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
                        if is_swap:                  # 1c: bowl_2(지시타깃) 접시 도달 판정
                            instructed = bool(env.env._eval_predicate(
                                ("on", "akita_black_bowl_2", "plate_1")))
                        if done or instructed:
                            break
                    z_hist.append(encode(obs))
            ok = bool(done)                          # LIBERO: done == success (bowl_1)
            if is_swap:
                instr, orig = bool(instructed), ok
                swap_instr.append(instr)
                swap_orig.append(orig)
                succ.append(instr)                   # primary SR = instructed(bowl_2)
                label = ("INSTRUCTED(bowl_2)" if instr
                         else "ORIG(bowl_1)" if orig else "neither")
                print(f"[task {tid}] ep {ep:2d} | {label} "
                      f"| steps {t} | 추론 {np.mean(infer_ms):.1f}ms", flush=True)
            else:
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
        # greppable 라인은 유지: swap 모드에선 X% = instructed-SR (기존 집계 호환)
        print(f"== task {tid} [{task.language[:50]}]: {sr:.0f}% "
              f"({int(np.sum(succ))}/{args.episodes})", flush=True)
        if is_swap:
            n = len(swap_instr)
            neither = [(not i) and (not o) for i, o in zip(swap_instr, swap_orig)]
            instr_sr = float(np.mean(swap_instr)) * 100
            orig_sr = float(np.mean(swap_orig)) * 100
            neither_sr = float(np.mean(neither)) * 100
            swap_results[tid] = (instr_sr, orig_sr, neither_sr)
            print(f"   [swap] instructed(bowl_2)={instr_sr:.0f}% "
                  f"({int(np.sum(swap_instr))}/{n}) | "
                  f"orig(bowl_1)={orig_sr:.0f}% ({int(np.sum(swap_orig))}/{n}) | "
                  f"neither={neither_sr:.0f}% ({int(np.sum(neither))}/{n})", flush=True)

    print(f"\n=== {args.suite} | 태스크당 {args.episodes} 롤아웃 | instr={args.instruction_mode} ===")
    for tid, sr in results.items():
        print(f"task {tid:2d}: {sr:5.1f}%  {suite.get_task(tid).language[:60]}")
    print(f"평균 성공률: {np.mean(list(results.values())):.1f}%")
    out = WS / "outputs" / "eval" / f"rollout_{args.suite}_{args.instruction_mode}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = ("\n".join(f"task{t}: {s:.1f}%" for t, s in results.items())
           + f"\nmean: {np.mean(list(results.values())):.1f}%\n")
    if is_swap:                                      # 1c: 3율 브레이크다운 추가 출력/저장
        print("--- swap (1c) instructed / orig / neither ---")
        for tid, (i_sr, o_sr, n_sr) in swap_results.items():
            print(f"task {tid:2d}: instructed {i_sr:5.1f}% | "
                  f"orig {o_sr:5.1f}% | neither {n_sr:5.1f}%")
        m_i = np.mean([v[0] for v in swap_results.values()])
        m_o = np.mean([v[1] for v in swap_results.values()])
        m_n = np.mean([v[2] for v in swap_results.values()])
        print(f"mean: instructed {m_i:.1f}% | orig {m_o:.1f}% | neither {m_n:.1f}%")
        txt += "".join(f"task{t}_swap: instructed {i:.1f}% orig {o:.1f}% "
                       f"neither {n:.1f}%\n"
                       for t, (i, o, n) in swap_results.items())
        txt += (f"mean_swap: instructed {m_i:.1f}% orig {m_o:.1f}% "
                f"neither {m_n:.1f}%\n")
    out.write_text(txt)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
