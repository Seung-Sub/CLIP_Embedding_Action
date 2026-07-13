"""W4 — 이미지·언어·행동 정렬 정량 분석 + 시각화 스위트 (재사용 가능).

cowork DIRECTIVE W4 (ledger '정렬의 의미론'):
  - QUANTITATIVE metrics가 판정 계층이다 (outputs/analysis/*.json).
  - VISUALIZATION은 예시(illustration)일 뿐 — 산점도의 시각 인상으로 정렬
    성공/실패를 판정하지 않는다. 모든 그림 캡션은 대응 정량 지표를 인용한다.

앵커별(anchor=clip|siglip2)로 phase1 DeltaAE(g: 행동청크→Δz)를 held-out(val)
쌍에 적용해 다음을 산출한다:

W4.1 정량 (판정 계층) → outputs/analysis/alignment_{anchor}.json
  · 정렬: per-sample cos(g(A,z), Δz) 히스토그램 통계 + ‖g‖/‖Δz‖ 노름비 분포
  · 검색: action→Δz / Δz→action top-1/5 (+ chance baseline)
  · CKA(선형): {Δz공간, g공간, 텍스트임베딩공간} 쌍별 + (양 앵커 존재 시)
    교차앵커 z_t공간 CKA (백본 스왑이 표현을 얼마나 바꾸는가)

W4.2 시각화 (예시) → outputs/presentation/FIG_analysis_{anchor}_*.png
  · 공유 2D 투영(PCA): z_t 궤적(태스크색/시간그라디언트), g 점군,
    모션문장 텍스트 오버레이
  · 접선투영 로컬뷰(Δz vs g 화살표)
  · UMAP: umap-learn 있으면 n_neighbors{15,50,200}×min_dist{0.1,0.5}×seed2 스윕
    → 대표 1장 + 부록(appendix) 저장. 없으면 PCA만 하고 UMAP 보류 리포트.

전부 CPU·캐시 임베딩 기반 (GPU 롤아웃 비간섭). 실행:
  CUDA_VISIBLE_DEVICES="" python src/analysis/alignment_report.py --anchor clip
  CUDA_VISIBLE_DEVICES="" python src/analysis/alignment_report.py --anchor siglip2
  CUDA_VISIBLE_DEVICES="" python src/analysis/alignment_report.py --cross   # 양 앵커 후
"""
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "src"))

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from core import chunkrep  # noqa: E402
from data.libero import LiberoDataset  # noqa: E402
from models.networks import DeltaAE  # noqa: E402

ANCHORS = {
    "clip":    {"config": "configs/phase1_libero.yaml",
                "ckpt":   "checkpoints/phase1_libero.pt"},
    "siglip2": {"config": "configs/phase1_libero_siglip2.yaml",
                "ckpt":   "checkpoints/phase1_libero_siglip2.pt"},
}

OUT_JSON = WS / "outputs" / "analysis"
OUT_FIG = WS / "outputs" / "presentation"
OUT_APPX = WS / "outputs" / "presentation" / "umap_sweep_appendix"
OUT_ARR = WS / "outputs" / "analysis" / "arrays"


# ═══════════════════════════════════════════════════════════════════════════
# 정량 유틸
# ═══════════════════════════════════════════════════════════════════════════

def dist_stats(x):
    """1D 분포 요약 (히스토그램 통계) — 판정 계층 수치."""
    x = np.asarray(x, dtype=np.float64)
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
            "q75": float(q[3]), "q95": float(q[4]), "n": int(len(x))}


def retrieval(Q, K, ks=(1, 5)):
    """query Q → key K 최근접 검색. top-k 정확도(%) + chance baseline(%)."""
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
    rank = (-(Qn @ Kn.T)).argsort(1)
    m = np.arange(len(Q))
    out = {}
    for k in ks:
        acc = float((rank[:, :k] == m[:, None]).any(1).mean() * 100)
        out[f"top{k}"] = acc
        out[f"chance_top{k}"] = float(100.0 * k / len(Q))
    return out


def linear_cka(X, Y):
    """선형 CKA (feature-space, sample-centered). 등방스케일·직교변환 불변.

    CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F · ||Yc^T Yc||_F).
    Xc, Yc: 표본(행) 평균 제거. NxN Gram 없이 dxd로 계산 (대규모 안전)."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy + 1e-12))


# ═══════════════════════════════════════════════════════════════════════════
# 로딩 / 쌍 구성 (학습과 동일한 seed 분할·starts)
# ═══════════════════════════════════════════════════════════════════════════

class _CacheAnchor:
    """이미지 임베딩 캐시 읽기 전용 스텁 — cache_key만 있으면 npz 히트(모델 미로드)."""
    def __init__(self, cache_key):
        self.cache_key = cache_key


def load_phase1(ckpt_path, device="cpu"):
    ck = torch.load(os.path.expanduser(ckpt_path), map_location="cpu",
                    weights_only=False)
    m = ck["config"]["model"]
    ae = DeltaAE(ck["action_dim"], ck["n_chunk"], ck["latent_dim"], m["hidden"],
                 m["layers"], m["dropout"], m.get("state_cond", True),
                 m.get("decoder_state_cond"), m.get("encoder_state_cond")
                 ).to(device).eval()
    ae.load_state_dict(ck["state_dict"])
    return ae, ck


def build_val_pairs(cfg, ck):
    """학습 스크립트(train_phase1)와 비트 동형인 seed 분할로 held-out 쌍 구성.

    반환: dict(Zt, Ztn, A, task_idx, tfrac, ep_key, task_names, span, tasks).
    임베딩은 캐시 히트(CPU), a_mean/a_std는 ckpt(train 통계) 사용."""
    ds = LiberoDataset(cfg)
    cache_key = ck["anchor"]["cache_key"]
    anchor_stub = _CacheAnchor(cache_key)
    files = ds.episode_files()
    seed = cfg["train"]["seed"]
    v = cfg["data"]["val_episodes"]
    perm = np.random.RandomState(seed).permutation(len(files))
    n_val = max(1, round(len(files) * v)) if v < 1 else int(v)
    val_ids = perm[:n_val]

    tasks = sorted({p for p, _ in files})
    task_of = {p: i for i, p in enumerate(tasks)}
    task_names = [ds.instruction((p, None)) for p in tasks]

    Zt_l, Ztn_l, A_l, tk_l, tf_l = [], [], [], [], []
    for i in val_ids:
        ep = files[i]
        acts = ds.load_actions(ep)
        T = len(acts)
        Z = ds.embeddings(anchor_stub, ep)          # 캐시 히트 (agentview)
        starts = list(range(0, T - ds.span, ds.stride))
        for t in starts:
            Zt_l.append(Z[t]); Ztn_l.append(Z[t + ds.span])
            A_l.append(ds.resample_chunk(acts[t:t + ds.span]).ravel())
            tk_l.append(task_of[ep[0]])
            tf_l.append(t / max(T - ds.span, 1))
    return {
        "Zt": np.asarray(Zt_l, np.float32),
        "Ztn": np.asarray(Ztn_l, np.float32),
        "A": np.asarray(A_l, np.float32),
        "task_idx": np.asarray(tk_l, np.int64),
        "tfrac": np.asarray(tf_l, np.float32),
        "task_names": task_names, "tasks": tasks,
        "span": ds.span, "n_chunk": ds.n_chunk,
    }


def compute_g(ae, ck, pairs, device="cpu"):
    """g(A, z_t) — 학습과 동일 정규화·표현. Δz = z_{t+span} - z_t."""
    n_chunk = ck["n_chunk"]; act_dim = ck["action_dim"]
    a_mean, a_std = ck["a_mean"], ck["a_std"]
    A = pairs["A"].reshape(len(pairs["A"]), n_chunk, act_dim)
    C = ((A - a_mean) / a_std).astype(np.float32)
    C = chunkrep.to_repr(C, ck.get("chunk_repr", "time"))
    with torch.no_grad():
        g = ae.g(torch.tensor(C, device=device),
                 torch.tensor(pairs["Zt"], device=device)).cpu().numpy()
    dz = pairs["Ztn"] - pairs["Zt"]
    return g.astype(np.float32), dz.astype(np.float32)


def text_embeddings(cfg, pairs, ck):
    """모션문장(v1) 텍스트 공간: 샘플별 배정 임베딩 (n, dim_text) + 카테고리.

    앵커 텍스트 인코더를 CPU에서 1회 forward (고유 문장 ~수백). 실패 시 None."""
    try:
        from core.anchor import get_anchor
        from data.motion_lang import MotionSentences
        anchor = get_anchor(cfg)                    # CUDA 숨김 시 CPU
        ms = MotionSentences(version=cfg["data"].get("motion_vocab", "v1"))
        A = pairs["A"].reshape(len(pairs["A"]), ck["n_chunk"], ck["action_dim"])
        sent_ids = ms.assign(A)
        sent_emb = ms.embed_all(anchor)             # (n_sent, dim_text)
        cats = np.array([ms.category_of_sent(s)[0] for s in sent_ids])
        return sent_emb[sent_ids].astype(np.float32), cats, sent_emb
    except Exception as e:                          # noqa: BLE001
        print(f"  [경고] 텍스트 임베딩 계산 실패 → 텍스트공간 CKA/오버레이 생략: {e}")
        return None, None, None


# ═══════════════════════════════════════════════════════════════════════════
# W4.1 리포트
# ═══════════════════════════════════════════════════════════════════════════

def quantitative(anchor, pairs, g, dz, text_per_sample):
    cos = (g * dz).sum(1) / (np.linalg.norm(g, axis=1)
                             * np.linalg.norm(dz, axis=1) + 1e-8)
    ratio = np.linalg.norm(g, axis=1) / (np.linalg.norm(dz, axis=1) + 1e-8)
    rep = {
        "anchor": anchor,
        "n_val_pairs": int(len(g)),
        "latent_dim": int(g.shape[1]),
        "alignment": {
            "cos_g_dz": dist_stats(cos),
            "norm_ratio_g_over_dz": dist_stats(ratio),
            "frac_cos_gt_0": float((cos > 0).mean()),
            "frac_cos_gt_0.5": float((cos > 0.5).mean()),
        },
        "retrieval": {
            "action_to_dz": retrieval(g, dz),       # query=g(행동) → key=Δz
            "dz_to_action": retrieval(dz, g),       # query=Δz → key=g(행동)
        },
        "cka_linear": {
            "dz_vs_g": linear_cka(dz, g),
        },
    }
    if text_per_sample is not None:
        rep["cka_linear"]["dz_vs_text"] = linear_cka(dz, text_per_sample)
        rep["cka_linear"]["g_vs_text"] = linear_cka(g, text_per_sample)
    return rep, cos, ratio


# ═══════════════════════════════════════════════════════════════════════════
# W4.2 시각화 (예시 — 캡션에 정량 지표 인용)
# ═══════════════════════════════════════════════════════════════════════════

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fit_pca(mats, k=2):
    X = np.concatenate([np.atleast_2d(m) for m in mats]).astype(np.float64)
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    basis = vt[:k]
    return mu, basis


def proj(X, mu, basis):
    return (np.atleast_2d(X) - mu) @ basis.T


def figures_pca(anchor, pairs, g, dz, rep, text_ps, text_cats, text_uni,
                task_lang=None, rng_seed=0, sub=1500):
    plt = _mpl()
    rng = np.random.RandomState(rng_seed)
    n = len(g)
    idx = rng.choice(n, min(sub, n), replace=False)
    Zt, Ztn = pairs["Zt"], pairs["Ztn"]
    task_idx, tfrac = pairs["task_idx"], pairs["tfrac"]

    # 공유 PCA basis: z_t·z_{t+span}·g점군(z_t+g)·텍스트 (동일 공간)
    pool = [Zt[idx], Ztn[idx], (Zt + g)[idx]]
    if text_uni is not None:
        pool.append(text_uni)
    if task_lang is not None:
        pool.append(task_lang)
    mu, basis = fit_pca(pool, 2)

    cos_mean = rep["alignment"]["cos_g_dz"]["mean"]
    cka_dg = rep["cka_linear"]["dz_vs_g"]
    ret = rep["retrieval"]["action_to_dz"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    cmap = plt.get_cmap("tab10")

    # (a) z_t 태스크색 + 태스크 언어 별
    ax = axes[0, 0]
    P = proj(Zt[idx], mu, basis)
    for ti in range(len(pairs["tasks"])):
        m = task_idx[idx] == ti
        ax.scatter(P[m, 0], P[m, 1], s=6, color=cmap(ti % 10), alpha=0.5)
    if task_lang is not None:
        L = proj(task_lang, mu, basis)
        ax.scatter(L[:, 0], L[:, 1], marker="*", s=320, c="k",
                   edgecolors="w", linewidths=1.2, zorder=5, label="task lang")
        ax.legend(loc="best", fontsize=9)
    ax.set_title(f"(a) z_t by task (10 tasks) [{anchor}]\n"
                 f"cite: cross-anchor/z-space CKA in JSON", fontsize=11)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # (b) z_t 시간 그라디언트
    ax = axes[0, 1]
    sc = ax.scatter(P[:, 0], P[:, 1], s=6, c=tfrac[idx], cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=ax, label="episode time fraction")
    ax.set_title("(b) z_t colored by time fraction\n"
                 "illustration of trajectory progression", fontsize=11)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # (c) g 점군(z_t+g) vs 실제 Δz 종점(z_t+Δz)
    ax = axes[1, 0]
    Pg = proj((Zt + g)[idx], mu, basis)
    Pd = proj(Ztn[idx], mu, basis)
    ax.scatter(Pd[:, 0], Pd[:, 1], s=6, c="#4477AA", alpha=0.5, label="z_t+Δz (true)")
    ax.scatter(Pg[:, 0], Pg[:, 1], s=6, c="#EE7733", alpha=0.5, label="z_t+g(A) (pred)")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"(c) g(A) point cloud vs true Δz endpoints\n"
                 f"QUANT: mean cos(g,Δz)={cos_mean:+.3f}, "
                 f"CKA(Δz,g)={cka_dg:.3f}, a→Δz top1={ret['top1']:.1f}%",
                 fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # (d) 모션문장 텍스트 오버레이 (같은 공유 공간)
    ax = axes[1, 1]
    ax.scatter(P[:, 0], P[:, 1], s=5, c="#CCCCCC", alpha=0.4, label="z_t")
    if text_uni is not None:
        T2 = proj(text_uni, mu, basis)
        ax.scatter(T2[:, 0], T2[:, 1], s=40, c="#CC3311", marker="^",
                   alpha=0.8, label="motion-sentence text")
        cka_dt = rep["cka_linear"].get("dz_vs_text")
        cap = (f"QUANT: CKA(Δz,text)={cka_dt:.3f}"
               if cka_dt is not None else "text-space CKA in JSON")
    else:
        cap = "text embedding unavailable (see JSON/report)"
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"(d) motion-sentence text overlay (shared space)\n{cap}",
                 fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle(f"FIG_analysis_{anchor}_pca — PCA is linear (arrows/global "
                 f"distances preserved). ILLUSTRATION ONLY; judgment = JSON metrics.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    p = OUT_FIG / f"FIG_analysis_{anchor}_pca.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  그림: {p}")

    # 접선투영 로컬뷰: Δz(파랑) vs g(주황) 화살표
    fig2, ax = plt.subplots(figsize=(9, 8))
    sel = rng.choice(n, 16, replace=False)
    for j in sel:
        p0 = proj(Zt[j], mu, basis)[0]
        pd = proj(Zt[j] + dz[j], mu, basis)[0]
        pg = proj(Zt[j] + g[j], mu, basis)[0]
        ax.annotate("", xy=pd, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color="#4477AA", lw=2))
        ax.annotate("", xy=pg, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color="#EE7733", lw=2, ls="--"))
        ax.scatter(*p0, c="k", s=25, zorder=5)
    ax.plot([], [], "-", color="#4477AA", lw=2, label="Δz (image change)")
    ax.plot([], [], "--", color="#EE7733", lw=2, label="g(A) (action→latent)")
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"FIG_analysis_{anchor}_tangent — local Δz vs g arrows (PCA)\n"
                 f"QUANT: mean cos(g,Δz)={cos_mean:+.3f} "
                 f"(median {rep['alignment']['cos_g_dz']['median']:+.3f}); "
                 f"norm ratio ‖g‖/‖Δz‖ mean="
                 f"{rep['alignment']['norm_ratio_g_over_dz']['mean']:.2f}",
                 fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    p2 = OUT_FIG / f"FIG_analysis_{anchor}_tangent.png"
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  그림: {p2}")
    return [str(p), str(p2)]


def figures_umap(anchor, pairs, g, dz, rep, rng_seed=0, sub=1500):
    """UMAP 스윕. umap-learn 없으면 (False, 사유) 반환 → PCA만."""
    try:
        import umap  # noqa: F401
    except Exception:                               # noqa: BLE001
        return False, "umap-learn 미설치 (import umap 실패) — UMAP 보류, PCA만 산출"
    import umap
    plt = _mpl()
    rng = np.random.RandomState(rng_seed)
    idx = rng.choice(len(g), min(sub, len(g)), replace=False)
    Zt = pairs["Zt"][idx]; tfrac = pairs["tfrac"][idx]; tk = pairs["task_idx"][idx]
    OUT_APPX.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    rep_fig = None
    for nn in (15, 50, 200):
        for md in (0.1, 0.5):
            for sd in (0, 1):
                emb = umap.UMAP(n_neighbors=nn, min_dist=md, random_state=sd,
                                n_components=2).fit_transform(Zt)
                fig, ax = plt.subplots(figsize=(7, 6))
                for ti in range(len(pairs["tasks"])):
                    m = tk == ti
                    ax.scatter(emb[m, 0], emb[m, 1], s=6, color=cmap(ti % 10), alpha=0.5)
                ax.set_title(f"UMAP z_t [{anchor}] nn={nn} md={md} seed={sd}\n"
                             "WARNING: UMAP does NOT preserve global distances; "
                             "apparent clusters can be artifacts. Judgment=JSON.",
                             fontsize=8)
                p = OUT_APPX / f"umap_{anchor}_nn{nn}_md{md}_s{sd}.png"
                fig.savefig(p, dpi=100, bbox_inches="tight")
                plt.close(fig)
                if nn == 50 and md == 0.1 and sd == 0:
                    rep_fig = str(p)
    if rep_fig:                                     # 대표 1장 복사
        import shutil
        dst = OUT_FIG / f"FIG_analysis_{anchor}_umap.png"
        shutil.copy(rep_fig, dst)
        print(f"  그림(UMAP 대표): {dst}")
        return True, str(dst)
    return True, "sweep done"


# ═══════════════════════════════════════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════════════════════════════════════

def run_anchor(anchor, do_umap=True):
    spec = ANCHORS[anchor]
    cfg = yaml.safe_load(open(WS / spec["config"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{anchor}] device={device} (GPU 숨김 시 CPU) — phase1 로드")
    ae, ck = load_phase1(WS / spec["ckpt"], device)
    print(f"[{anchor}] held-out 쌍 구성 (seed={cfg['train']['seed']}) ...")
    pairs = build_val_pairs(cfg, ck)
    print(f"[{anchor}] val pairs = {len(pairs['A'])}  dim={ck['latent_dim']}")
    g, dz = compute_g(ae, ck, pairs, device)
    text_ps, text_cats, text_uni = text_embeddings(cfg, pairs, ck)
    # 태스크 언어 임베딩 (캐시): 각 태스크 대표
    try:
        from core.anchor import get_anchor
        anc = get_anchor(cfg)
        ds = LiberoDataset(cfg)
        task_lang = np.stack([ds.instruction_embedding(anc, (p, None))
                              for p in pairs["tasks"]]).astype(np.float32)
    except Exception as e:                          # noqa: BLE001
        print(f"  [경고] 태스크 언어 임베딩 생략: {e}")
        task_lang = None

    rep, cos, ratio = quantitative(anchor, pairs, g, dz, text_ps)
    rep["stored_ckpt_metrics"] = ck.get("metrics")  # 교차검증용

    OUT_JSON.mkdir(parents=True, exist_ok=True)
    jp = OUT_JSON / f"alignment_{anchor}.json"
    jp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[{anchor}] JSON: {jp}")

    figs = figures_pca(anchor, pairs, g, dz, rep, text_ps, text_cats, text_uni,
                       task_lang)
    umap_ok, umap_msg = (figures_umap(anchor, pairs, g, dz, rep)
                         if do_umap else (False, "skipped"))
    print(f"[{anchor}] UMAP: {umap_msg}")
    rep["_umap"] = {"available": umap_ok, "note": umap_msg}
    rep["_figures"] = figs
    jp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    # 교차앵커용 배열 저장 (행 정렬 동일 — 동일 files/seed/starts)
    OUT_ARR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_ARR / f"arr_{anchor}.npz",
                        Zt=pairs["Zt"], dz=dz, g=g,
                        task_idx=pairs["task_idx"], tfrac=pairs["tfrac"])
    return rep


def run_cross():
    """교차앵커 z_t/Δz/g 공간 CKA (백본 스왑 영향). 양 앵커 배열 필요."""
    pc = OUT_ARR / "arr_clip.npz"; ps = OUT_ARR / "arr_siglip2.npz"
    if not (pc.exists() and ps.exists()):
        print("교차앵커 CKA: 양 앵커 배열 없음 — 먼저 각 앵커 실행 필요")
        return None
    a = np.load(pc); b = np.load(ps)
    assert len(a["Zt"]) == len(b["Zt"]), "행 정렬 불일치 (files/seed/starts 달라짐?)"
    out = {
        "note": "동일 files/seed/starts로 행 1:1 정렬된 CLIP↔SigLIP2 표현 비교",
        "n_pairs": int(len(a["Zt"])),
        "cka_zt_clip_vs_siglip2": linear_cka(a["Zt"], b["Zt"]),
        "cka_dz_clip_vs_siglip2": linear_cka(a["dz"], b["dz"]),
        "cka_g_clip_vs_siglip2": linear_cka(a["g"], b["g"]),
    }
    jp = OUT_JSON / "alignment_cross_anchor.json"
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"교차앵커 CKA JSON: {jp}\n{json.dumps(out, indent=2)}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# W4 v2 — "beyond-cosine" battery (판정 계층). 모든 수치는 CPU·캐시 기반.
#   1 Probing + control-task selectivity + MDL codelength (Hewitt&Liang;Voita&Titov)
#   2 Distance correlation Δz↔true action-delta, granularity-split (Székely 2007)
#   3 CKA hardening: debiased CKA + orthogonal-Procrustes + PWCCA (Davari;Morcos)
#   4 Cross-modal: mutual-kNN(Huh 2024)+modality gap(Liang 2022)+align/unif(Wang&Isola)
#   5 Geometry: IsoScore(Rudman 2022)+effective rank(Roy&Vetterli)+TwoNN(Facco 2017)
# 주의: raw CKA/코사인은 조작 가능(caveat) — 항상 2차 지표와 병기, control 대비 선택도로 판정.
# ═══════════════════════════════════════════════════════════════════════════

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402


# ── 모션 라벨 (참 action 청크에서 파생; data.motion_lang.chunk_category 재사용) ──

def motion_labels(pairs, ck):
    """참(raw) action 청크 → 모션 속성 라벨. 축/방향/크기/그리퍼는
    학습 정렬 타깃과 동일한 chunk_category 규약(지배축·MAG_BOUNDARY)으로 파생."""
    from data.motion_lang import AXES, chunk_category
    n = len(pairs["A"])
    A = pairs["A"].reshape(n, ck["n_chunk"], ck["action_dim"])
    cum = A[:, :, :6].sum(1)                       # (n,6) 누적 EE 변위(참)
    axis6 = np.empty(n, np.int64); sgn = np.empty(n, np.int64)
    magbin = np.empty(n, np.int64); gripc = np.empty(n, np.int64)
    for i, ch in enumerate(A):
        cat, g = chunk_category(ch)                # "z-|L", g∈{0,1,2}
        base = cat.split("|")[0]
        axis6[i] = AXES.index(base[:-1]); sgn[i] = 0 if base[-1] == "+" else 1
        magbin[i] = 1 if cat.endswith("L") else 0; gripc[i] = g
    ar = np.arange(n)
    axis3 = np.abs(cum[:, :3]).argmax(1)           # 지배 병진축 (x/y/z)
    dir_dom = (cum[ar, axis3] < 0).astype(np.int64)
    dom_mag = np.abs(cum[ar, axis3]).astype(np.float64)
    return {
        "axis3": axis3, "dir_dom": dir_dom, "mag_bin": magbin,
        "grip3": gripc, "grip_event": (gripc > 0).astype(np.int64),
        "axis6": axis6, "dom_mag": dom_mag,
        "action_cum6": cum.astype(np.float64), "action_full": pairs["A"],
    }


# ── 1. Probing + control + MDL ──────────────────────────────────────────────

def _mdl_online(Xs, y, n_classes, seed, blocks=(0.1, 0.25, 0.5, 1.0), eps=0.05):
    """Prequential(online) codelength [bits] (Voita&Titov 2020): 접두부로 학습→
    다음 블록을 부호화. 용량-안정. 첫 블록은 균일부호(log2 K).
    확률 스무딩 p'=(1-eps)p + eps/K: 고차원 소표본 프로브의 과확신 오분류로 인한
    codelength 발산 방지(균일성분 혼합부호) → 샘플당 비용 ≤ log2(K/eps)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y)); Xs = Xs[perm]; y = y[perm]
    n = len(y); idx = sorted(set([max(1, int(f * n)) for f in blocks] + [n]))
    code = idx[0] * np.log2(n_classes)
    for a, b in zip([idx[0]] + idx[:-1], idx):
        if b <= a:
            continue
        if len(np.unique(y[:a])) < 2:
            code += (b - a) * np.log2(n_classes); continue
        clf = LogisticRegression(max_iter=500, C=1.0).fit(Xs[:a], y[:a])
        proba = clf.predict_proba(Xs[a:b])
        col = {c: i for i, c in enumerate(clf.classes_)}
        p = np.full(b - a, eps / n_classes)
        for j, yt in enumerate(y[a:b]):
            if yt in col:
                p[j] = (1 - eps) * proba[j, col[yt]] + eps / n_classes
        code += float(-np.log2(p).sum())
    return float(code)


def probe_clf(X, y, seed=0, do_mdl=True):
    """선형 프로브(로지스틱). 실태스크 acc + 다수결 baseline + control-task(라벨 셔플)
    acc → 선택도(selectivity)=acc-control. + MDL codelength(real vs control)."""
    K = int(len(np.unique(y)))
    scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25,
                                          random_state=seed, stratify=y)
    acc = float(accuracy_score(yte, LogisticRegression(max_iter=300, C=1.0)
                               .fit(Xtr, ytr).predict(Xte)))
    _, cnts = np.unique(ytr, return_counts=True)
    maj = float(cnts.max() / cnts.sum())
    yc = np.random.RandomState(seed + 1).permutation(y)   # random-label control
    Xt2, Xe2, yct, yce = train_test_split(Xs, yc, test_size=0.25,
                                          random_state=seed, stratify=yc)
    acc_c = float(accuracy_score(yce, LogisticRegression(max_iter=300, C=1.0)
                                 .fit(Xt2, yct).predict(Xe2)))
    out = {"n_classes": K, "acc": acc, "majority_baseline": maj,
           "acc_control": acc_c, "selectivity_acc": acc - acc_c}
    if do_mdl:
        mr = _mdl_online(Xs, y, K, seed); mc = _mdl_online(Xs, yc, K, seed)
        uni = len(y) * np.log2(K)
        out.update({"mdl_bits_real": mr, "mdl_bits_control": mc,
                    "mdl_uniform_bits": uni,
                    "compression_real": uni / mr, "compression_control": uni / mc,
                    "mdl_selectivity_bits": mc - mr})
    return out


def probe_reg(X, y, seed=0):
    """선형 회귀(Ridge) R² + control(셔플) R² → 선택도."""
    scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=seed)
    r2 = float(r2_score(yte, Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)))
    yc = np.random.RandomState(seed + 1).permutation(y)
    Xt2, Xe2, yct, yce = train_test_split(Xs, yc, test_size=0.25, random_state=seed)
    r2c = float(r2_score(yce, Ridge(alpha=1.0).fit(Xt2, yct).predict(Xe2)))
    return {"r2": r2, "r2_control": r2c, "selectivity_r2": r2 - r2c}


def probing_battery(g, dz, zt, lab, task_idx):
    """Δz / g(A) 가 모션 축·방향·크기·그리퍼를 control 이상으로 선형 인코딩하는가.
    task_idx 프로브(장면 언어 프록시)는 zt/dz에서만 — motion 라벨과 달리 간접."""
    clf_targets = {"axis3(x/y/z)": lab["axis3"], "dir_dom(±)": lab["dir_dom"],
                   "mag_bin(S/L)": lab["mag_bin"], "grip_event": lab["grip_event"],
                   "grip3(none/close/open)": lab["grip3"]}
    feats = {"dz": dz, "g": g}
    out = {"note": "selectivity=acc-control(random-label,Hewitt&Liang);"
                   " MDL codelength bits(Voita&Titov). 라벨은 참 action청크 파생."}
    out["classification"] = {}
    for tname, y in clf_targets.items():
        out["classification"][tname] = {f: probe_clf(feats[f], y) for f in feats}
    out["regression_dom_magnitude"] = {f: probe_reg(feats[f], lab["dom_mag"])
                                        for f in feats}
    # 언어/과제 함량 (프록시, 간접): z_t 및 Δz → task_idx(10-way)
    out["task_identity_proxy"] = {
        "note": "UNVERIFIED-as-language: task_idx is scene-level id, a coarse proxy"
                " for scene language content, not a motion-semantic label.",
        "zt": probe_clf(zt, task_idx, do_mdl=False),
        "dz": probe_clf(dz, task_idx, do_mdl=False)}
    return out


# ── 2. Distance correlation (Székely 2007), granularity-split ────────────────

def dcorr(X, Y, max_n=2000, seed=0):
    """distance correlation ∈[0,1] (0 ⇔ 독립). 비선형·비등차원 의존 포착.
    n>max_n 이면 고정 시드로 서브샘플 (O(n²) 메모리)."""
    X = np.atleast_2d(X).astype(np.float64); Y = np.atleast_2d(Y).astype(np.float64)
    if X.shape[0] > max_n:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_n, replace=False)
        X = X[idx]; Y = Y[idx]
    a = squareform(pdist(X)); b = squareform(pdist(Y))
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean(); vx = (A * A).mean(); vy = (B * B).mean()
    den = np.sqrt(vx * vy)
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(den)) if den > 0 else 0.0


def ridge_multi_r2(X, Y, seed=0):
    """다출력 Ridge: Y(참 action-delta)를 X(Δz/g)로 예측한 held-out 평균 R²(선형)."""
    Xs = StandardScaler().fit_transform(X)
    Xtr, Xte, Ytr, Yte = train_test_split(Xs, Y, test_size=0.25, random_state=seed)
    return float(r2_score(Yte, Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte),
                          multioutput="variance_weighted"))


def dcor_battery(g, dz, lab):
    cum6 = lab["action_cum6"]; full = lab["action_full"].astype(np.float64)
    coarse = lab["mag_bin"] == 1; fine = lab["mag_bin"] == 0
    grip = lab["grip_event"] == 1
    out = {"note": "dCor(Székely): 0⇔독립, 비선형·비등차원. R²는 선형 비교치. "
                   "granularity: coarse=대변위(mag L), fine=소변위(mag S).",
           "target": "true action-delta = per-chunk cumulative EE Δpose (6-dim)"}
    for name, X in (("dz", dz), ("g", g)):
        out[name] = {
            "dcor_vs_cum6_all": dcorr(X, cum6),
            "dcor_vs_fullchunk112": dcorr(X, full),
            "linear_R2_cum6_from_X": ridge_multi_r2(X, cum6),
            "dcor_vs_cum6_coarse": dcorr(X[coarse], cum6[coarse]),
            "dcor_vs_cum6_fine": dcorr(X[fine], cum6[fine]),
            "dcor_vs_gripcol_gripevents": (
                dcorr(X[grip], full.reshape(len(full), -1)[grip][:, 6::7])
                if grip.sum() > 50 else None),
            "n_coarse": int(coarse.sum()), "n_fine": int(fine.sum()),
            "n_gripevent": int(grip.sum())}
    return out


# ── 3. CKA hardening: debiased CKA + Procrustes + PWCCA ──────────────────────

def _hsic1(K, L):
    """Unbiased HSIC (Song 2012) — debiased CKA용."""
    n = K.shape[0]
    K = K.copy(); L = L.copy(); np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
    o = np.ones(n)
    t1 = np.trace(K @ L)
    t2 = (o @ K @ o) * (o @ L @ o) / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (o @ K @ L @ o)
    return (t1 + t2 - t3) / (n * (n - 3))


def cka_debiased(X, Y, max_n=2000, seed=0):
    """Debiased linear CKA (Nguyen 2021). raw CKA의 표본편향·outlier 민감성 완화."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if X.shape[0] > max_n:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_n, replace=False)
        X = X[idx]; Y = Y[idx]
    X = X - X.mean(0); Y = Y - Y.mean(0)
    K = X @ X.T; L = Y @ Y.T
    hxy = _hsic1(K, L); hxx = _hsic1(K, K); hyy = _hsic1(L, L)
    den = np.sqrt(max(hxx, 0) * max(hyy, 0))
    return float(hxy / den) if den > 0 else float("nan")


def _pca_to(X, d):
    Xc = X - X.mean(0)
    if X.shape[1] <= d:
        return Xc
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:d].T


def procrustes_dist(X, Y, max_n=4000, seed=0):
    """직교 Procrustes 거리 ∈[0,√2]: 0=회전/반사 무시 시 동일. 등차원 필요→PCA로
    공통 min차원 축소. Frobenius 정규화 후 nuclear norm 로 계산."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if X.shape[0] > max_n:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_n, replace=False)
        X = X[idx]; Y = Y[idx]
    d = min(X.shape[1], Y.shape[1])
    Xp = _pca_to(X, d); Yp = _pca_to(Y, d)
    Xp = Xp / (np.linalg.norm(Xp) + 1e-12); Yp = Yp / (np.linalg.norm(Yp) + 1e-12)
    s = np.linalg.svd(Xp.T @ Yp, compute_uv=False).sum()   # nuclear norm
    return float(np.sqrt(max(2.0 - 2.0 * s, 0.0)))


def pwcca(X, Y, max_n=4000, seed=0, var=0.99):
    """PWCCA (Morcos 2018): 투영가중 CCA 유사도 ∈[0,1]. 비등차원 허용, CKA의 2차 지표."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if X.shape[0] > max_n:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_n, replace=False)
        X = X[idx]; Y = Y[idx]
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    Ux, sx, _ = np.linalg.svd(Xc, full_matrices=False)
    Uy, sy, _ = np.linalg.svd(Yc, full_matrices=False)
    kx = int(np.searchsorted(np.cumsum(sx**2) / np.sum(sx**2), var) + 1)
    ky = int(np.searchsorted(np.cumsum(sy**2) / np.sum(sy**2), var) + 1)
    Ux = Ux[:, :kx]; Uy = Uy[:, :ky]
    u, rho, _ = np.linalg.svd(Ux.T @ Uy)
    rho = np.clip(rho, 0, 1)
    H = Ux @ u                                     # X 정준변량
    alpha = np.abs(H.T @ Xc).sum(1)
    alpha = alpha / (alpha.sum() + 1e-12)
    m = min(len(rho), len(alpha))
    return float((alpha[:m] * rho[:m]).sum())


def cka_hardening(spaces):
    """{Δz,g,z_t,text} 쌍별: raw CKA(참고), debiased CKA, Procrustes 거리, PWCCA."""
    keys = list(spaces); out = {
        "caveat": "raw linear CKA is manipulable & outlier-sensitive (Davari 2022);"
                  " report debiased CKA + Procrustes/PWCCA together, never CKA alone.",
        "pairs": {}}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if spaces[a] is None or spaces[b] is None:
                continue
            X, Y = spaces[a], spaces[b]
            out["pairs"][f"{a}__{b}"] = {
                "cka_linear_raw": linear_cka(X, Y),
                "cka_debiased": cka_debiased(X, Y),
                "procrustes_dist": procrustes_dist(X, Y),
                "pwcca": pwcca(X, Y),
                "dims": [int(X.shape[1]), int(Y.shape[1])]}
    return out


# ── 4. Cross-modal alignment (Huh 2024; Liang 2022; Wang&Isola 2020) ─────────

def _norm(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def mutual_knn(A, B, k=10, max_n=2500, seed=0):
    """상호 kNN 정렬 (Platonic RH, Huh 2024): 쌍 대응 두 공간의 이웃집합 겹침 평균 ∈[0,1]."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    if A.shape[0] > max_n:
        idx = np.random.RandomState(seed).choice(A.shape[0], max_n, replace=False)
        A = A[idx]; B = B[idx]

    def knn(M):
        Mn = _norm(M); S = Mn @ Mn.T; np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, 1)[:, :k]
    na, nb = knn(A), knn(B)
    return float(np.mean([len(set(na[i]) & set(nb[i])) / k for i in range(len(A))]))


def crossmodal(zt, dz, g, text_ps, task_lang_ps, max_n=2500, seed=0):
    """이미지↔언어 정렬 품질(백본 랭킹용). text=모션문장(샘플별, 공유공간).
    modality gap/align/unif 은 동일 공간(=동일 차원)에서만."""
    out = {"note": "text = per-sample motion-sentence embedding in the shared "
                   "image-text space (the phase-1 alignment target). mutual-kNN "
                   "model-agnostic; modality gap & align/unif need shared dim.",
           "k_for_mutual_knn": 10}
    if text_ps is None:
        out["status"] = "UNVERIFIED: text embeddings unavailable"
        return out

    def unif(M):
        Mn = _norm(M)
        if Mn.shape[0] > max_n:
            Mn = Mn[np.random.RandomState(seed).choice(Mn.shape[0], max_n, False)]
        return float(np.log(np.exp(-2.0 * pdist(Mn, "sqeuclidean")).mean()))

    def block(img, txt, name):
        d = {"mutual_knn_img_text": mutual_knn(img, txt)}
        if img.shape[1] == txt.shape[1]:
            In, Tn = _norm(img), _norm(txt)
            d["modality_gap"] = float(np.linalg.norm(In.mean(0) - Tn.mean(0)))
            d["alignment_pos_pair"] = float(((In - Tn) ** 2).sum(1).mean())
            d["uniformity_img"] = unif(img); d["uniformity_text"] = unif(txt)
        else:
            d["note"] = f"{name}: img/text dims differ → gap/align/unif skipped"
        return d
    out["zt_vs_motiontext"] = block(zt, text_ps, "zt")
    out["dz_vs_motiontext"] = block(dz, text_ps, "dz")
    out["g_vs_motiontext"] = block(g, text_ps, "g")
    if task_lang_ps is not None:
        out["zt_vs_taskinstruction"] = block(zt, task_lang_ps, "zt-task")
    return out


# ── 5. Geometry: IsoScore + effective rank + TwoNN ───────────────────────────

def effective_rank(X):
    """effective rank = exp(H(정규화 특이값)) (Roy&Vetterli 2007)."""
    Xc = np.asarray(X, np.float64) - np.asarray(X, np.float64).mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / (s.sum() + 1e-12); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def isoscore(X, max_n=4000, seed=0):
    """IsoScore ∈[0,1] (Rudman 2022): 공분산 고유값 정규화 후 등방성 결손 → 이용 차원 비율."""
    X = np.asarray(X, np.float64)
    if X.shape[0] > max_n:
        X = X[np.random.RandomState(seed).choice(X.shape[0], max_n, False)]
    X = X - X.mean(0)
    lam = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None)
    d = len(lam)
    lam = lam * np.sqrt(d) / (np.linalg.norm(lam) + 1e-12)   # ‖lam‖=√d
    delta = np.linalg.norm(lam - 1.0) / np.sqrt(2.0 * (d - np.sqrt(d)))
    k = d - delta ** 2 * (d - np.sqrt(d))
    return float((k ** 2 - d) / (d ** 2 - d))


def twonn(X, frac=0.9, max_n=4000, seed=0):
    """intrinsic dimension (TwoNN, Facco 2017): μ=r2/r1 의 CDF 선형회귀 기울기."""
    X = np.asarray(X, np.float64)
    if X.shape[0] > max_n:
        X = X[np.random.RandomState(seed).choice(X.shape[0], max_n, False)]
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dd, _ = nn.kneighbors(X)
    mu = dd[:, 2] / np.maximum(dd[:, 1], 1e-12)
    mu = np.sort(mu[mu > 1 + 1e-9]); N = len(mu)
    F = np.arange(1, N + 1) / (N + 1)
    keep = int(frac * N)
    x = np.log(mu[:keep]); y = -np.log(1.0 - F[:keep])
    return float((x * y).sum() / (x * x).sum())


def geometry_battery(zt, dz, g):
    out = {"note": "IsoScore/eff-rank/TwoNN with anisotropy prior (Ethayarajh 2019);"
                   " high raw dim ≠ more space used."}
    for name, X in (("z_t", zt), ("dz", dz), ("g", g)):
        out[name] = {"ambient_dim": int(X.shape[1]),
                     "isoscore": isoscore(X),
                     "effective_rank": effective_rank(X),
                     "twonn_intrinsic_dim": twonn(X)}
    return out


# ── v2 실행 ─────────────────────────────────────────────────────────────────

def run_v2(anchor):
    import warnings
    warnings.filterwarnings("ignore")              # 프로브 수렴 경고(비판정) 소거
    spec = ANCHORS[anchor]
    cfg = yaml.safe_load(open(WS / spec["config"]))
    print(f"[v2:{anchor}] phase1 로드 + held-out 쌍 재구성 (seed={cfg['train']['seed']})")
    ae, ck = load_phase1(WS / spec["ckpt"], "cpu")
    pairs = build_val_pairs(cfg, ck)
    g, dz = compute_g(ae, ck, pairs, "cpu")
    zt = pairs["Zt"]
    text_ps, _, _ = text_embeddings(cfg, pairs, ck)
    try:
        from core.anchor import get_anchor
        anc = get_anchor(cfg); ds = LiberoDataset(cfg)
        tl = np.stack([ds.instruction_embedding(anc, (p, None))
                       for p in pairs["tasks"]]).astype(np.float32)
        task_lang_ps = tl[pairs["task_idx"]]
    except Exception as e:                          # noqa: BLE001
        print(f"  [경고] task 언어 임베딩 생략: {e}"); task_lang_ps = None
    lab = motion_labels(pairs, ck)

    print(f"[v2:{anchor}] 1/5 probing (+control+MDL) …")
    prob = probing_battery(g, dz, zt, lab, pairs["task_idx"])
    print(f"[v2:{anchor}] 2/5 distance correlation …")
    dc = dcor_battery(g, dz, lab)
    print(f"[v2:{anchor}] 3/5 CKA hardening …")
    hard = cka_hardening({"dz": dz, "g": g, "z_t": zt, "text": text_ps})
    print(f"[v2:{anchor}] 4/5 cross-modal alignment …")
    cm = crossmodal(zt, dz, g, text_ps, task_lang_ps)
    print(f"[v2:{anchor}] 5/5 geometry …")
    geo = geometry_battery(zt, dz, g)

    rep = {"anchor": anchor, "n_val_pairs": int(len(g)),
           "latent_dim": int(g.shape[1]),
           "text_available": text_ps is not None,
           "probing": prob, "distance_correlation": dc,
           "cka_hardening": hard, "crossmodal_alignment": cm, "geometry": geo}
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    jp = OUT_JSON / f"alignment_v2_{anchor}.json"
    jp.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[v2:{anchor}] JSON: {jp}")
    return rep


def run_cross_v2():
    """교차앵커(z_t/Δz/g) CKA hardening — 조작 불가한 2차 지표로 백본 스왑 비교."""
    pc = OUT_ARR / "arr_clip.npz"; ps = OUT_ARR / "arr_siglip2.npz"
    if not (pc.exists() and ps.exists()):
        print("cross-v2: 양 앵커 배열 없음 — 먼저 v1 각 앵커 실행 필요"); return None
    a = np.load(pc); b = np.load(ps)
    assert len(a["Zt"]) == len(b["Zt"]), "행 정렬 불일치"
    out = {"note": "CLIP↔SigLIP2 (동일 files/seed/starts 1:1). raw CKA 단독 금지 —"
                   " debiased CKA+Procrustes+PWCCA 병기 (Davari;Morcos).",
           "n_pairs": int(len(a["Zt"]))}
    for sp in ("Zt", "dz", "g"):
        out[sp] = {"cka_linear_raw": linear_cka(a[sp], b[sp]),
                   "cka_debiased": cka_debiased(a[sp], b[sp]),
                   "procrustes_dist": procrustes_dist(a[sp], b[sp]),
                   "pwcca": pwcca(a[sp], b[sp])}
    jp = OUT_JSON / "alignment_v2_cross_anchor.json"
    jp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"cross-v2 JSON: {jp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=list(ANCHORS), default=None)
    ap.add_argument("--all", action="store_true", help="모든 앵커 + 교차앵커")
    ap.add_argument("--cross", action="store_true", help="교차앵커 CKA만")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="beyond-cosine battery (--anchor 지정 또는 --all)")
    ap.add_argument("--cross-v2", action="store_true", help="교차앵커 hardening만")
    args = ap.parse_args()

    if args.cross_v2:
        run_cross_v2(); return
    if args.v2:
        if args.all:
            for a in ANCHORS:
                run_v2(a)
            run_cross_v2(); return
        if args.anchor:
            run_v2(args.anchor); return
        ap.error("--v2 with --anchor {clip,siglip2} 또는 --all")
    if args.cross:
        run_cross(); return
    if args.all:
        for a in ANCHORS:
            run_anchor(a, do_umap=not args.no_umap)
        run_cross(); return
    if args.anchor:
        run_anchor(args.anchor, do_umap=not args.no_umap); return
    ap.error("--anchor {clip,siglip2} | --all | --cross | --v2 중 하나")


if __name__ == "__main__":
    main()
