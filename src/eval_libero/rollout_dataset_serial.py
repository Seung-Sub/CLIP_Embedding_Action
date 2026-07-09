"""LIBERO 다단계 연쇄(월드모델 블라인드 구간) 대화형 뷰어.

n스텝(재계획 윈도우 단위) 동안 실측 이미지를 다시 인코딩하지 않고, 정책이 낸 ζ̂를
그대로 "예측된 미래 델타"로 써서 세계모델을 전진시킨다:

  z_hat(t+H) = z_cur + ζ̂        (한 스텝 상상)

n=0이면 항상 실측(=기존 rollout_dataset.py와 동일한 예측), n이 커질수록 더 오래
카메라 없이 "눈감고" 진행한다. n의 최댓값은 그 에피소드를 끝까지 덮는 윈도우 수
(=전체 데이터셋 길이)까지 설정 가능하다.

과거 액션(a_emb 계산용)은 항상 GT를 사용한다(개루프 진단 — 정책 자체의 액션 품질이
아니라 "월드모델 전진 능력"만 분리해서 본다. 정책 출력 품질까지 같이 보려면
eval_libero/rollout_sim.py의 폐루프 평가를 참조).

사용 (clip_libero env, 데스크톱 세션):
  python src/eval_libero/rollout_dataset_serial.py
  python src/eval_libero/rollout_dataset_serial.py --config configs/phase2_libero.yaml
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse
import collections
import os

import matplotlib
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider

from core import chunkrep
from core.clip_wrapper import ClipWrapper
from data.libero import LiberoDataset
from eval_libero.rollout_dataset import load_models  # NOTE: this module sets a CJK font.family
                                                       # as a side effect of import - override below

FS = 3.0
matplotlib.rcParams.update({"font.family": ["sans-serif"], "font.size": 10 * FS,
                            "axes.unicode_minus": False})   # after the import above, on purpose

DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def rollout_chain(ae, policy, acts, Z, a_mean, a_std, repr_kind, ds, H,
                  n_chain, lang, device):
    """반환: pred_base(항상 실측, H스텝 재계획), pred_chain(n청크 블라인드, span단위 전진),
    drift(청크별 cos), n_chunks.

    ★ ζ̂(zeta)는 g의 학습 타깃과 동일하게 "span(16스텝) 전체"에 대한 델타이므로,
    월드모델 전진(z_cur += Δ)도 반드시 span 단위로 해야 한다 — H(8)단위로 더하면
    스텝마다 델타를 2배 과다적용하게 된다.
    ★★ 전진에 ζ̂를 직접 쓰지 않는다 — flow 정책의 ζ̂는 실제 Δz와의 코사인 정합도가
    구조적으로 낮다(~0.1, 학습로그의 "잠재 cos" 실측치). 대신 "디코딩한 행동을 다시
    g로 인코딩"한 값을 쓴다 — g 자체의 정렬도(align_cos ~0.5~0.65)가 훨씬 신뢰할 만하고,
    이게 곧 실험1(A1)/실험2(B1)에서 확인한 "g가 실제 정렬 담당자"라는 결론과도 일치한다.
    그래서 블라인드 체인은 span-그리드로, 기준(항상실측) 곡선만 기존처럼 H-그리드
    재계획을 유지한다(더 촘촘해서 비교 기준선으로 더 정확함).
    """
    T, span = len(acts), ds.span

    def norm(a):
        return ((a - a_mean) / a_std).astype(np.float32)

    lang_t = torch.tensor(lang[None], device=device) if lang is not None else None

    def infer(zp_np, zc_np, t):
        past = chunkrep.to_repr(norm(ds.resample_chunk(acts[t - span:t])), repr_kind)[None]
        zp = torch.tensor(zp_np[None], device=device)
        zc = torch.tensor(zc_np[None], device=device)
        a_emb = ae.g(torch.tensor(past, device=device), zp)
        toks = [zp, zc, a_emb] + ([lang_t] if lang_t is not None else [])
        zeta = policy(torch.stack(toks, dim=1))
        ahat_t = ae.h(zeta, zc)                                      # (1, span, act_dim) 정규화·repr 공간
        ahat = chunkrep.from_repr(ahat_t.cpu().numpy()[0], repr_kind) * a_std + a_mean
        return ahat, zc, ahat_t

    def world_step(zc, ahat_t):
        """디코딩한 행동을 다시 g로 인코딩 — ζ̂보다 신뢰할 만한 전진용 Δz."""
        return ae.g(ahat_t, zc)

    # ---- 기준: 항상 실측, H스텝 재계획 (기존 rollout_dataset.py와 동일 촘촘함) ----
    pred_base = np.full_like(acts, np.nan)
    z_hist_len = span // H + 1
    hist_base = collections.deque(
        [Z[min(k * H, T - 1)].astype(np.float32) for k in range(z_hist_len)],
        maxlen=z_hist_len)
    with torch.no_grad():
        t = span
        while t + H <= T or t < T:
            n_exec = min(H, T - t)
            if n_exec <= 0:
                break
            ahat_b, _, _ = infer(hist_base[0], hist_base[-1], t)
            pred_base[t:t + n_exec] = ahat_b[:n_exec]
            hist_base.append(Z[min(t + H, T - 1)].astype(np.float32))
            t += H

    # ---- 블라인드 체인: span 단위 전진 (ζ̂의 실제 시간 스케일과 일치) ----
    n_chunks = max(0, (T - span) // span)
    pred_chain = np.full_like(acts, np.nan)
    drift = np.full(n_chunks, np.nan)
    z_prev_c, z_cur_c = Z[0].astype(np.float32), Z[min(span, T - 1)].astype(np.float32)
    with torch.no_grad():
        for k in range(n_chunks):
            t = span + k * span
            n_exec = min(span, T - t)
            ahat_c, zc_t, ahat_t = infer(z_prev_c, z_cur_c, t)
            pred_chain[t:t + n_exec] = ahat_c[:n_exec]
            if k < n_chain:
                delta_hat = world_step(zc_t, ahat_t)              # g(디코딩한 행동, z_cur) — 신뢰할 만한 Δ
                z_next = (zc_t + delta_hat).cpu().numpy()[0]      # ★월드모델 span단위 전진
                real_next = Z[min(t + span, T - 1)]
                num = float((z_next * real_next).sum())
                den = float(np.linalg.norm(z_next) * np.linalg.norm(real_next) + 1e-8)
                drift[k] = num / den
            else:
                z_next = Z[min(t + span, T - 1)].astype(np.float32)   # 실측 재동기화
            z_prev_c, z_cur_c = z_cur_c, z_next.astype(np.float32)

    return pred_base, pred_chain, drift, n_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero_nowrist.yaml"))
    ap.add_argument("--snapshot", default=None,
                    help="save current state to png instead of opening a window (headless check)")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    if args.snapshot:
        matplotlib.use("Agg")

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model and embedding cache...")
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam, *_) = load_models(cfg, device)   # *_ = F3 obs (미사용)
    ds = LiberoDataset(cfg)
    clip = ClipWrapper()

    eps = ds.episode_files()
    tasks = sorted({p for p, _ in eps})
    by_task = {p: [e for e in eps if e[0] == p] for p in tasks}
    names = [ds.instruction((p, None)) for p in tasks]
    pre = os.path.commonprefix(names)

    state = {"task": args.task_id, "ep": args.episode, "n": args.n, "H": 8}
    cache = {}
    SPAN = ds.span
    GLOBAL_MAX_CHUNKS = max(1, max((len(ds.load_actions(e)) - SPAN) // SPAN
                                   for e in eps))
    print(f"Global max blind chunks across dataset: {GLOBAL_MAX_CHUNKS}")

    def get_episode(ti, ei):
        ep = by_task[tasks[ti]][ei % len(by_task[tasks[ti]])]
        key = (ti, ei)
        if key not in cache:
            acts = ds.load_actions(ep)
            Z = ds.embeddings(clip, ep)
            lang = ds.instruction_embedding(clip, ep) if use_lang else None
            cache[key] = (acts, Z, lang, ep)
        return cache[key]

    fig = plt.figure(figsize=(26, 16))
    fig.canvas.manager.set_window_title("LIBERO Multi-step Chain Viewer") if fig.canvas.manager else None
    ax_grid = [fig.add_axes([0.04 + 0.34 * c, 0.74 - 0.23 * r, 0.30, 0.20])
              for r in range(4) for c in range(2)][:7]

    fig.text(0.72, 0.955, "Task (LIBERO-Spatial)", fontsize=9 * FS, weight="bold")
    ax_radio = fig.add_axes([0.71, 0.57, 0.27, 0.36], frameon=False)
    labels = [f"{i}: {n[len(pre):][:32] or n[:32]}" for i, n in enumerate(names)]
    radio = RadioButtons(ax_radio, labels)
    for lb in radio.labels:
        lb.set_fontsize(6 * FS)

    ax_ep = fig.add_axes([0.72, 0.44, 0.25, 0.03])
    s_ep = Slider(ax_ep, "Episode", 0, 49, valinit=0, valstep=1)
    s_ep.label.set_fontsize(8 * FS); s_ep.valtext.set_fontsize(8 * FS)

    fig.text(0.72, 0.395, "n (blind chunks, 0.8s each)", fontsize=6.5 * FS)
    n_slot = fig.add_axes([0.72, 0.35, 0.25, 0.03])
    n_holder = {"slider": Slider(n_slot, "", 0, GLOBAL_MAX_CHUNKS,
                                 valinit=min(args.n, GLOBAL_MAX_CHUNKS), valstep=1)}
    n_holder["slider"].valtext.set_fontsize(8 * FS)

    info = fig.text(0.72, 0.22, "", fontsize=7.5 * FS, color="#222222")
    fig.text(0.72, 0.03,
             "Blue=GT  Green dashed=always-real (baseline)  Orange=n-chunk blind (world-model rollforward)",
             fontsize=6.5 * FS, color="#444444")

    def redraw():
        ti, ei, n = state["task"], int(state["ep"]), int(state["n"])
        acts, Z, lang, ep = get_episode(ti, ei)
        H = state["H"]
        # figure out this episode's own chunk ceiling BEFORE running the chain,
        # so a short episode actively pulls the slider value down (range stays
        # at GLOBAL_MAX_CHUNKS so a longer episode can still use the full range).
        n_chunks_ep = max(0, (len(acts) - ds.span) // ds.span)
        if n > n_chunks_ep:
            n = n_chunks_ep
            state["n"] = n
            sl = n_holder["slider"]
            sl.eventson = False
            sl.set_val(n)
            sl.eventson = True

        pred_base, pred_chain, drift, n_chunks = rollout_chain(
            ae, policy, acts, Z, a_mean, a_std, repr_kind, ds, H, n, lang, device)

        T = len(acts)
        tt = np.arange(T) / 20.0
        mae_b = np.nanmean(np.abs(pred_base - acts))
        mae_c = np.nanmean(np.abs(pred_chain - acts))
        for d in range(act_dim):
            a = ax_grid[d]; a.clear()
            a.plot(tt, acts[:, d], color="#4477AA", lw=1.8, label="GT")
            a.plot(tt, pred_base[:, d], color="#228833", lw=1.2, ls="--", label="always-real")
            a.plot(tt, pred_chain[:, d], color="#EE7733", lw=1.4, label=f"n={n} blind")
            a.set_title(DIM_NAMES[d], fontsize=8 * FS)
            a.tick_params(labelsize=6 * FS)
            a.grid(color="#EEEEEE", lw=0.4)
        ax_grid[0].legend(fontsize=6.5 * FS, loc="upper right")

        info.set_text(f"task {ti} episode {ei} (T={T}, {n_chunks} chunks max)\n"
                      f"MAE always-real: {mae_b:.3f}\nMAE n={n} blind: {mae_c:.3f}\n"
                      f"final drift: {drift[n-1]:.3f}" if n > 0 else
                      f"task {ti} episode {ei} (T={T}, {n_chunks} chunks max)\n"
                      f"MAE always-real: {mae_b:.3f}")
        fig.canvas.draw_idle()

    def on_task(label):
        state["task"] = int(label.split(":")[0]); redraw()

    def on_ep(v):
        state["ep"] = int(v); redraw()

    def on_n(v):
        state["n"] = int(v); redraw()

    radio.on_clicked(on_task)
    s_ep.on_changed(on_ep)
    n_holder["slider"].on_changed(on_n)

    redraw()
    if args.snapshot:
        fig.savefig(args.snapshot, dpi=110, bbox_inches="tight")
        print(f"saved: {args.snapshot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
