"""잠재공간 맵핑 시각화 — LIBERO phase1(delta-AE) 기반, 대화형 창.

하나의 CLIP 잠재공간(768) 위 요소들을 PCA(선형투영이라 벡터 화살표가 보존됨)로
2D/3D 산점도에 함께 표시한다:

  ● 3인칭(agentview) 전 z_t / 후 z_{t+16}      + Δz 화살표 (실선, 파랑)
  ▲ 액션청크 g(A, z_t)                          → z_t에서 출발하는 화살표 (점선, 주황)
                                                  — Δz 화살표와 방향·크기 직접 비교
  ■ 그리퍼(eye_in_hand) 전/후                   + 그 델타 화살표 (실선, 초록)
  ★ 언어 cmd (CLIP 텍스트 임베딩)
  · 선택 에피소드의 z 궤적 (회색 실선, 문맥용)

  ▲ 동일 ζ 화살표를 그리퍼 전 지점에서도 표시 (점선, 주황) — 그리퍼 델타와 비교

우측 컨트롤: 태스크 목록(라디오) / 에피소드·시작 시점 슬라이더 /
토글 [3D] = 2D↔3D 전환, [이미지 영역 확대] = 모달리티 갭 무시하고 3인칭 영역 확대.

사용 (clip_libero 또는 clip env, 데스크톱 세션):
  python src/eval_libero/latent_mapping.py                     # 창 실행
  python src/eval_libero/latent_mapping.py --snapshot out.png  # 창 없이 현재 상태 저장(점검용)
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse
import os

import h5py
import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import font_manager

for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(_f)
    except FileNotFoundError:
        pass
_cjk = next((f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name),
            "sans-serif")
FS = 3.0                                     # 전체 글씨 배율
matplotlib.rcParams.update({"font.family": [_cjk, "sans-serif"],
                            "font.size": 10 * FS,
                            "axes.unicode_minus": False})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import CheckButtons, RadioButtons, Slider  # noqa: E402

from core import chunkrep  # noqa: E402
from core.clip_wrapper import ClipWrapper  # noqa: E402
from data.libero import LiberoDataset  # noqa: E402
from models.networks import DeltaAE  # noqa: E402

C_AGENT, C_ACT, C_WRIST, C_LANG = "#4477AA", "#EE7733", "#228833", "#CC3311"
TAB = plt.get_cmap("tab10")


def load_phase1(cfg, device):
    ck = torch.load(os.path.expanduser(cfg["phase1_ckpt"]),
                    map_location="cpu", weights_only=False)
    p1 = ck["config"]["model"]
    ae = DeltaAE(ck["action_dim"], ck["n_chunk"], p1["latent_dim"], p1["hidden"],
                 p1["layers"], p1["dropout"], p1.get("state_cond", True),
                 p1.get("decoder_state_cond"), p1.get("encoder_state_cond")
                 ).to(device).eval()
    ae.load_state_dict(ck["state_dict"])
    return ae, ck["a_mean"], ck["a_std"], ck.get("chunk_repr", "time")


class LatentMapper:
    def __init__(self, cfg, device):
        self.device = device
        self.ds = LiberoDataset(cfg)
        self.wrist_cam = cfg["data"].get("wrist_camera", "eye_in_hand_rgb")
        self.clip = ClipWrapper()
        self.ae, self.a_mean, self.a_std, self.repr = load_phase1(cfg, device)
        self.span = self.ds.span

        eps = self.ds.episode_files()
        self.tasks = sorted({p for p, _ in eps})            # 태스크 = hdf5 파일
        self.by_task = {p: [e for e in eps if e[0] == p] for p in self.tasks}
        self.names = [self.ds.instruction(( p, None)) for p in self.tasks]
        self.lang = np.stack([self.ds.instruction_embedding(self.clip, (p, None))
                              for p in self.tasks])          # (10, 768)
        self._cache = {}
        self._fit_pca()

    # ---------- 데이터 ----------

    def episode(self, ti, ei):
        """(태스크 idx, 에피소드 idx) → (Z, Zw, acts). 임베딩은 npz 캐시 재사용."""
        ei = ei % len(self.by_task[self.tasks[ti]])
        ep = self.by_task[self.tasks[ti]][ei]
        key = (ti, ei)
        if key not in self._cache:
            Z = self.ds.embeddings(self.clip, ep)
            Zw = self.ds.embeddings(self.clip, ep, self.wrist_cam)
            acts = self.ds.load_actions(ep)
            self._cache[key] = (Z, Zw, acts)
        return self._cache[key]

    def frame(self, ti, ei, t, camera=None):
        """단일 프레임만 hdf5에서 부분 읽기 (상세 보기용)."""
        ei = ei % len(self.by_task[self.tasks[ti]])
        path, demo = self.by_task[self.tasks[ti]][ei]
        with h5py.File(path, "r") as h:
            return h[f"data/{demo}/obs/{camera or self.ds.camera}"][t]

    def g_vec(self, z_t, acts, t):
        """액션청크 → 잠재 ζ = g(A_{t:t+span}, z_t)."""
        seg = self.ds.resample_chunk(acts[t:t + self.span])
        seg = ((seg - self.a_mean) / self.a_std).astype(np.float32)
        seg = chunkrep.to_repr(seg, self.repr)
        with torch.no_grad():
            zeta = self.ae.g(torch.tensor(seg[None], device=self.device),
                             torch.tensor(z_t[None], device=self.device))
        return zeta.cpu().numpy()[0]

    # ---------- PCA (선형 → 화살표 보존) ----------

    def _fit_pca(self):
        pool = [self.lang]
        for ti in range(len(self.tasks)):
            Z, Zw, acts = self.episode(ti, 0)
            pool += [Z[::5], Zw[::5]]
            for t in range(0, len(acts) - self.span, self.span * 2):
                pool.append((Z[t] + self.g_vec(Z[t], acts, t))[None])
        X = np.concatenate(pool).astype(np.float64)
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.basis = vt[:3]                                  # (3, 768)

    def proj(self, X, dim):
        return (np.atleast_2d(X) - self.mu) @ self.basis[:dim].T


# ---------- 그리기 ----------

def draw_arrow(ax, p0, p1, color, ls="-", lw=1.8, label=None):
    d = p1 - p0
    if len(p0) == 3:
        ax.quiver(*p0, *d, color=color, lw=lw, linestyle=ls,
                  arrow_length_ratio=0.12, label=label)
    else:
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    linestyle=ls, shrinkA=0, shrinkB=0))
        if label:                                            # 범례용 대리 라인
            ax.plot([], [], color=color, ls=ls, lw=lw, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--snapshot", default=None,
                    help="창 대신 현재 상태를 png로 저장 (헤드리스 점검용)")
    ap.add_argument("--detail", action="store_true",
                    help="상세 보기 창을 켠 상태로 시작")
    args = ap.parse_args()
    if args.snapshot:
        matplotlib.use("Agg")

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("phase1 모델·임베딩 캐시 로드 중...")
    M = LatentMapper(cfg, device)
    state = {"task": 0, "ep": 0, "frac": 0.0,
             "d3": False, "zoom": False, "detail": args.detail}

    fig = plt.figure(figsize=(26, 16))           # 4배 글씨에 맞춘 대형 캔버스
    fig.canvas.manager.set_window_title("LIBERO 잠재공간 맵핑 (phase1)") \
        if fig.canvas.manager else None

    def make_ax():
        return fig.add_axes([0.04, 0.07, 0.60, 0.88],
                            projection="3d" if state["d3"] else None)

    ax = make_ax()

    # ---- 우측 컨트롤 ----
    fig.text(0.66, 0.955, "태스크 (LIBERO-Spatial)", fontsize=9 * FS, weight="bold")
    ax_radio = fig.add_axes([0.65, 0.50, 0.34, 0.42], frameon=False)
    # 태스크명 공통 접두어(예: "pick up the black bowl ")는 떼고 구별부만 표시
    pre = os.path.commonprefix(M.names)
    labels = [f"{i}: {n[len(pre):][:32] or n[:32]}" for i, n in enumerate(M.names)]
    radio = RadioButtons(ax_radio, labels)
    for lb in radio.labels:
        lb.set_fontsize(7 * FS)
    ax_ep = fig.add_axes([0.72, 0.42, 0.23, 0.03])
    s_ep = Slider(ax_ep, "에피소드", 0, 49, valinit=0, valstep=1)
    ax_t = fig.add_axes([0.72, 0.35, 0.23, 0.03])
    s_t = Slider(ax_t, "시작 시점", 0.0, 1.0, valinit=0.0)
    for s in (s_ep, s_t):
        s.label.set_fontsize(8 * FS)
        s.valtext.set_fontsize(8 * FS)
    info_text = fig.text(0.72, 0.30, "", fontsize=8 * FS, color="#222222")
    ax_chk = fig.add_axes([0.70, 0.10, 0.28, 0.16], frameon=False)
    chk = CheckButtons(ax_chk, ["3D", "이미지 영역 확대", "상세 보기(이미지·액션궤적)"],
                       [False, False, args.detail])
    for lb in chk.labels:
        lb.set_fontsize(8 * FS)
    fig.text(0.65, 0.02,
             "화살표: 파랑 실선=Δz(이미지 전→후) · 주황 점선=g(액션청크)\n"
             "초록=그리퍼 델타 · ★=언어 cmd · 회색=z 궤적",
             fontsize=6.5 * FS, color="#444444")

    # ---- 상세 보기: 전/후 이미지 4장 + 3D 액션 궤적 (별도 창) ----
    detail = {"fig": None}

    def redraw_detail(ti, ei, t, t2, acts):
        if not state["detail"]:
            if detail["fig"] is not None:
                plt.close(detail["fig"])
                detail["fig"] = None
            return
        new = detail["fig"] is None
        if new:
            detail["fig"] = plt.figure(figsize=(22, 12))
        f2 = detail["fig"]
        f2.clf()
        gs = f2.add_gridspec(2, 4, height_ratios=[1, 1.5],
                             hspace=0.15, wspace=0.08)
        t2c = min(t2, len(acts) - 1)
        shots = [(M.frame(ti, ei, t), "3인칭 전 (t)"),
                 (M.frame(ti, ei, t2c), "3인칭 후 (t+16)"),
                 (M.frame(ti, ei, t, M.wrist_cam), "그리퍼 전"),
                 (M.frame(ti, ei, t2c, M.wrist_cam), "그리퍼 후")]
        for k, (img, ttl) in enumerate(shots):
            axi = f2.add_subplot(gs[0, k])
            axi.imshow(img[::-1])                # 사람 눈 기준 방향 (표시 전용 플립)
            axi.set_title(ttl, fontsize=6 * FS)
            axi.axis("off")
        ax3 = f2.add_subplot(gs[1, :], projection="3d")
        pos = np.cumsum(acts[:, :3], axis=0)     # Δpos 누적 ≈ EEF 경로
        ax3.plot(*pos.T, color="#BBBBBB", lw=1.5)
        ax3.scatter(*pos[0][:, None], color="k", s=80, marker="o")
        seg = pos[t:t2c + 1]
        ax3.plot(*seg.T, color=C_ACT, lw=4.5)
        ax3.scatter(*seg[0][:, None], color=C_ACT, s=90, marker="o",
                    facecolors="none", linewidths=2.5)
        ax3.scatter(*seg[-1][:, None], color=C_ACT, s=110, marker="^")
        ax3.set_title("전체 액션 궤적 (Δpos 누적, ●=시작) — 주황=현재 청크 (○→▲)",
                      fontsize=6 * FS)
        ax3.tick_params(labelsize=4.5 * FS)
        if new and not args.snapshot:
            f2.show()
        f2.canvas.draw_idle()

    # ---- 렌더 ----
    def redraw():
        nonlocal ax
        ax.remove()
        ax = make_ax()
        dim = 3 if state["d3"] else 2

        ti, ei = state["task"], int(state["ep"])
        Z, Zw, acts = M.episode(ti, ei)
        T = len(acts)
        t = int(round(state["frac"] * max(T - M.span - 1, 0)))
        t2 = t + M.span
        zeta = M.g_vec(Z[t], acts, t)

        traj = M.proj(Z, dim)
        ax.plot(*traj.T, color="#BBBBBB", lw=0.8, alpha=0.8, zorder=1)

        a0, a1 = M.proj(Z[t], dim)[0], M.proj(Z[t2], dim)[0]
        w0, w1 = M.proj(Zw[t], dim)[0], M.proj(Zw[t2], dim)[0]
        gv = M.proj(Z[t] + zeta, dim)[0]
        gw = M.proj(Zw[t] + zeta, dim)[0]        # 동일 ζ를 그리퍼 전 지점에서도
        L = M.proj(M.lang[ti], dim)[0]

        ax.scatter(*a0[:, None], color=C_AGENT, s=240, marker="o",
                   facecolors="none", linewidths=2.5, label="3인칭 전 z_t")
        ax.scatter(*a1[:, None], color=C_AGENT, s=240, marker="o",
                   label="3인칭 후 z_{t+16}")
        ax.scatter(*w0[:, None], color=C_WRIST, s=200, marker="s",
                   facecolors="none", linewidths=2.5, label="그리퍼 전")
        ax.scatter(*w1[:, None], color=C_WRIST, s=200, marker="s",
                   label="그리퍼 후")
        ax.scatter(*L[:, None], color=C_LANG, s=600, marker="*",
                   edgecolors="k", linewidths=0.8, label="언어 cmd")
        draw_arrow(ax, a0, a1, C_AGENT, "-", 3.0, label="Δz (이미지 전→후)")
        draw_arrow(ax, a0, gv, C_ACT, "--", 3.0, label="g(액션청크) — Δz와 비교")
        draw_arrow(ax, w0, w1, C_WRIST, "-", 2.6, label="그리퍼 Δ")
        draw_arrow(ax, w0, gw, C_ACT, "--", 2.6)  # 동일 ζ, 그리퍼 시점 비교용

        if state["zoom"]:                          # 모달리티 갭 무시, 3인칭 영역만
            pts = np.vstack([traj, gv[None]])
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.12 * (hi - lo + 1e-9)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            if dim == 3:
                ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])

        cos = float(np.dot(zeta, Z[t2] - Z[t]) /
                    (np.linalg.norm(zeta) * np.linalg.norm(Z[t2] - Z[t]) + 1e-8))
        info_text.set_text(f"t = {t} → {t2}  (T={T})\n"
                           f"cos(g, Δz) = {cos:+.3f}")
        ax.legend(fontsize=6 * FS, loc="best")
        redraw_detail(ti, ei, t, t2, acts)

        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if dim == 3:
            ax.set_zlabel("PC3")
        else:
            ax.grid(color="#EEEEEE", lw=0.5)
        fig.canvas.draw_idle()

    # ---- 콜백 ----
    def on_radio(label):
        state["task"] = int(label.split(":")[0]); redraw()

    def on_ep(v):
        state["ep"] = int(v); redraw()

    def on_t(v):
        state["frac"] = float(v); redraw()

    def on_chk(label):
        key = {"3D": "d3", "이미지 영역 확대": "zoom",
               "상세 보기(이미지·액션궤적)": "detail"}[label]
        state[key] = not state[key]
        redraw()

    radio.on_clicked(on_radio)
    s_ep.on_changed(on_ep)
    s_t.on_changed(on_t)
    chk.on_clicked(on_chk)

    redraw()
    if args.snapshot:
        fig.savefig(args.snapshot, dpi=120, bbox_inches="tight")
        print(f"저장: {args.snapshot}")
        if detail["fig"] is not None:
            p = args.snapshot.replace(".png", "_detail.png")
            detail["fig"].savefig(p, dpi=120, bbox_inches="tight")
            print(f"저장: {p}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
