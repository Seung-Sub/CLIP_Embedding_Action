"""week2 dual-policy / dynamic-weighting 측정 공용 유틸 (M-A/M-B/M-C).

DESIGN_dualpolicy_dynamic_weighting_v1 사전 측정 — 전부 CPU, 기존 캐시+ckpt만 사용.
ckpt = W-C 표준화 dual (phase{1,2}_libero_dualstream_wrist_std.pt, N3/N4).
방법론 관례: week0/w0_common + wc_g2c_ablation 동형 (split 재현, DummyAnchor 캐시 전용).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

WS = Path(os.path.expanduser("~/clip_ws"))
sys.path.insert(0, str(WS / "src"))
sys.path.insert(0, str(WS / "scratchpad" / "week0"))

from w0_common import DummyAnchor, r2_pooled, split_ids   # noqa: E402
from core import chunkrep                                  # noqa: E402
from data.libero import LiberoDataset                      # noqa: E402
from models.networks import DualDeltaAE                    # noqa: E402
from models.policy import FlowPolicy, build_policy_from_cfg  # noqa: E402

torch.set_num_threads(int(os.environ.get("WK2_THREADS", "12")))

P1 = WS / "checkpoints" / "phase1_libero_dualstream_wrist_std.pt"
P2 = WS / "checkpoints" / "phase2_libero_dualstream_wrist_std.pt"
OUT = WS / "outputs" / "week2_dualdyn"
OUT.mkdir(parents=True, exist_ok=True)
GRIP = 6

# 그립 온셋-상대 bin (frame 단위, 20Hz): −16..+16 을 4프레임 폭 8칸 + far 양측
BIN_EDGES = list(range(-16, 17, 4))          # [-16,-12,...,16]
BIN_NAMES = (["far_pre"]
             + [f"[{a},{b})" for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
             + ["far_post", "no_grasp"])


def load_dual_std():
    """W-C 표준화 dual phase1(ae)+phase2(policy) 재구성 — wc_g2c_ablation 동형."""
    ck1 = torch.load(str(P1), map_location="cpu", weights_only=False)
    assert ck1.get("dual_stream")
    p1 = ck1["config"]
    dm, dw = ck1["dim_main"], ck1["dim_wrist"]
    ae = DualDeltaAE(ck1["action_dim"], ck1["n_chunk"], dm, dw,
                     p1["model"]["hidden"], p1["model"]["layers"],
                     p1["model"]["dropout"], p1["model"].get("state_cond", True),
                     stream_standardize=p1["model"].get("stream_standardize", False))
    ae.load_state_dict(ck1["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    assert ae.stream_standardize

    ck2 = torch.load(str(P2), map_location="cpu", weights_only=False)
    cfg2 = ck2["config"]
    m_cfg = cfg2["module"]
    assert m_cfg.get("wrist_cond_sig")
    use_lang = bool(m_cfg.get("lang_token", False))
    model = build_policy_from_cfg(m_cfg, n_tokens=4 + int(use_lang),
                                  latent_dim=dm + dw)
    assert isinstance(model, FlowPolicy) and model.flow_dim == dm + dw
    model.load_state_dict(ck2["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ck1, ae, ck2, cfg2, model, use_lang


def grasp_onset(acts):
    """에피소드 그립 온셋 프레임 = 최초의 '지속 닫힘 명령' 시작.

    LIBERO gripper 커맨드(dim 6): −1 열림 / +1 닫힘. 온셋 = g>0.5 이고
    이후 5프레임 중 4프레임 이상 닫힘 유지가 처음 성립하는 t. 없으면 −1."""
    g = acts[:, GRIP]
    close = g > 0.5
    for t in np.flatnonzero(close):
        if close[t:t + 5].mean() >= 0.8:
            return int(t)
    return -1


def bin_index(offset, has_grasp):
    """frame offset(start − t_on) → bin id (BIN_NAMES 인덱스)."""
    if not has_grasp:
        return len(BIN_NAMES) - 1
    if offset < BIN_EDGES[0]:
        return 0
    if offset >= BIN_EDGES[-1]:
        return len(BIN_NAMES) - 2
    return 1 + (int(offset) - BIN_EDGES[0]) // 4


def build_split(cfg2, ck1, split="val"):
    """policy 샘플 + 메타 (ep/task/frame/grasp offset/bin) — 배열 dict 반환.

    배열: Zp Zc Zn Ap Af Zwp Zwc Zwn Ws (build_policy_samples dual+wrist_cond 순서),
    Cf(정규화 미래청크 flatten), meta: ep_id, task_id, frame, onset, offset, bin,
    g_prev(직전 실행 gripper 커맨드 — 롤아웃 인과 가용), g_closing(최근 8f 내 부호전환)."""
    ds = LiberoDataset(cfg2)
    files = ds.episode_files()
    val_ids, tr_ids = split_ids(len(files), cfg2["train"]["seed"],
                                cfg2["data"]["val_episodes"])
    ids = {"val": val_ids, "train": tr_ids,
           "all": np.concatenate([val_ids, tr_ids])}[split]
    sel = [files[i] for i in ids]
    dm, dw = ck1["dim_main"], ck1["dim_wrist"]
    main_a = DummyAnchor(ck1["anchor"]["cache_key"], dm)
    wrist_a = DummyAnchor(ck1["anchor_wrist"]["cache_key"], dw)
    stride = cfg2["data"].get("stride", 2)
    eps = ds.build_policy_samples(main_a, sel, stride=stride,
                                  wrist_anchor=wrist_a,
                                  wrist_cond_anchor=main_a)
    names = ["Zp", "Zc", "Zn", "Ap", "Af", "Zwp", "Zwc", "Zwn", "Ws"]
    out = {k: np.concatenate([e[i] for e in eps])
           for i, k in enumerate(names)}

    # 태스크 id = 파일 stem (10개), 메타는 starts 재현으로 정렬 보장
    task_names = sorted({p.stem for p, _ in sel})
    t_of = {n: i for i, n in enumerate(task_names)}
    ep_id, task_id, frame, onset = [], [], [], []
    g_prev, g_closing = [], []
    span = ds.span
    for e_i, ep in enumerate(sel):
        acts = ds.load_actions(ep)
        T = len(acts)
        starts = list(range(0, T - span, stride))
        assert len(starts) == len(eps[e_i][0]), "starts 재현 불일치"
        t_on = grasp_onset(acts)
        g = acts[:, GRIP]
        for s in starts:
            ep_id.append(e_i)
            task_id.append(t_of[ep[0].stem])
            frame.append(s)
            onset.append(t_on)
            gp = g[s - 1] if s > 0 else g[0]
            g_prev.append(float(gp))
            lo = max(s - 8, 0)
            g_closing.append(bool((g[lo:s + 1] > 0.5).any()
                                  and (g[lo:s + 1] < -0.5).any()))
    meta = {"ep_id": np.array(ep_id), "task_id": np.array(task_id),
            "frame": np.array(frame), "onset": np.array(onset)}
    meta["offset"] = meta["frame"] - meta["onset"]
    meta["has_grasp"] = meta["onset"] >= 0
    meta["bin"] = np.array([bin_index(o, h) for o, h in
                            zip(meta["offset"], meta["has_grasp"])])
    meta["g_prev"] = np.array(g_prev)
    meta["g_closing"] = np.array(g_closing)
    meta["task_names"] = task_names
    meta["n_eps"] = len(sel)

    a_mean, a_std = ck1["a_mean"], ck1["a_std"]
    n_chunk, act_dim = ck1["n_chunk"], ck1["action_dim"]
    repr_kind = ck1.get("chunk_repr", "time")
    assert repr_kind == "time"

    def norm(A):
        a = ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std
             ).astype(np.float32)
        return chunkrep.to_repr(a, repr_kind).reshape(len(A), -1)

    out["Cf"] = norm(out["Af"])
    out["Cp"] = norm(out["Ap"])
    return out, meta, ds


def policy_zeta(model, ae, arrs, use_lang, ds, cfg2, ck1, files_meta=None,
                lang_arr=None):
    """W-C 정책 ζ̂ + oracle 표준화 ζ_gt — wc_g2c 동형 (deterministic, source=past)."""
    dm, dw = ck1["dim_main"], ck1["dim_wrist"]
    dc = dm + dw
    n = len(arrs["Cf"])
    zeta_np = np.empty((n, dc), np.float32)
    zeta_gt = np.empty((n, dc), np.float32)

    def _pad(t):
        return t if t.shape[-1] == dc else \
            torch.nn.functional.pad(t, (0, dc - t.shape[-1]))

    bs = 1024
    with torch.no_grad():
        for i in range(0, n, bs):
            j = slice(i, min(i + bs, n))
            t = {k: torch.from_numpy(arrs[k][j]) for k in
                 ("Zp", "Zc", "Zn", "Cp", "Zwp", "Zwc", "Zwn", "Ws")}
            a_emb = ae.encode(t["Cp"].view(len(t["Cp"]), ck1["n_chunk"],
                                           ck1["action_dim"]),
                              t["Zp"], t["Zwp"])
            toks = [_pad(t["Zp"]), _pad(t["Zc"]), a_emb, _pad(t["Ws"])]
            if use_lang:
                toks.append(_pad(torch.from_numpy(lang_arr[j])))
            gen = torch.Generator(); gen.manual_seed(0)
            zeta_np[j] = model(torch.stack(toks, 1), generator=gen).numpy()
            zeta_gt[j] = torch.cat(ae.std_dz(t["Zn"] - t["Zc"],
                                             t["Zwn"] - t["Zwc"]), 1).numpy()
    return zeta_np, zeta_gt


def lang_embeddings(ds, main_anchor, sel_files, eps_len):
    L = [ds.instruction_embedding(main_anchor, f) for f in sel_files]
    return np.concatenate([np.repeat(L[i][None], eps_len[i], 0)
                           for i in range(len(sel_files))]).astype(np.float32)


def decode_batched(ae, zeta, Zc, Zwc, bs=2048):
    n = len(zeta)
    outs = []
    with torch.no_grad():
        for i in range(0, n, bs):
            j = slice(i, min(i + bs, n))
            a = ae.decode(torch.from_numpy(zeta[j]),
                          torch.from_numpy(Zc[j]), torch.from_numpy(Zwc[j]))
            outs.append(a.numpy())
    return np.concatenate(outs).reshape(n, -1)


def per_dim_r2(y, p, n_chunk, act_dim, mask=None):
    y = y.reshape(-1, n_chunk, act_dim)
    p = p.reshape(-1, n_chunk, act_dim)
    if mask is not None:
        y, p = y[mask], p[mask]
    out = []
    for k in range(act_dim):
        yk = y[:, :, k].ravel().astype(np.float64)
        pk = p[:, :, k].ravel().astype(np.float64)
        dev = ((yk - yk.mean()) ** 2).sum()
        out.append(float(1 - ((yk - pk) ** 2).sum() / (dev + 1e-12)))
    return out
