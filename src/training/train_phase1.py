"""Phase 1 — 액션청크<->Δz 결합 AE (VITA 동형) 학습.

  구조/손실: configs/delta_ae.yaml 참조
  평가(held-out 에피소드): 디코더 R², cycle R², 양방향 검색 top-1/5

사용 (clipx env):
  python src/training/train_phase1.py            # 본 학습 (configs/phase1.yaml)
  python src/training/train_phase1.py --smoke    # 코드 점검 (2 eps)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from core import chunkrep
from core.anchor import get_anchor
from data import get_dataset
from models.networks import DeltaAE, DualDeltaAE

WS = Path(__file__).resolve().parents[2]
CFG_PATH = WS / "configs" / "phase1.yaml"


def r2(y, yhat):
    dev = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / (dev + 1e-12))


def retrieval(Q, K):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
    rank = (-(Qn @ Kn.T)).argsort(1)
    m = np.arange(len(Q))
    return (float((rank[:, 0] == m).mean() * 100),
            float((rank[:, :5] == m[:, None]).any(1).mean() * 100))


def apply_override(cfg, kv):
    """'train.lr=3e-4' 형식 오버라이드를 cfg dict에 적용."""
    key, val = kv.split("=", 1)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(val)


def run_dual(cfg, args):
    """Phase1 dual-stream 변위 AE 학습 (model.dual_stream=true 전용).

    카메라별 분리 앵커·인코더: agentview(anchor) + wrist(anchor_wrist, 예 DINOv3).
      g_main(a,z_main)→ζ_main≈Δz_main / g_wrist(a,z_wrist)→ζ_wrist≈Δz_wrist,
      h([ζ_main;ζ_wrist],[z_main;z_wrist])→액션청크 (DualDeltaAE).
    단일 스트림 경로(main)와 완전 분리 — 단일 런은 이 함수를 절대 타지 않음."""
    t_cfg, m_cfg, w = cfg["train"], cfg["model"], cfg["loss"]
    rng = np.random.RandomState(t_cfg["seed"])
    torch.manual_seed(t_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = get_dataset(cfg)
    files = ds.episode_files()
    if args.smoke:
        files = files[:2]
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    n_val = 1 if args.smoke else (max(1, round(len(files) * v)) if v < 1 else int(v))
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    print(f"[dual] episodes: train {len(tr_ids)} / val {len(val_ids)}")

    clip_main = get_anchor(cfg)
    clip_wrist = get_anchor({"anchor": cfg["anchor_wrist"]})
    dim_main, dim_wrist = clip_main.dim, clip_wrist.dim
    assert dim_main and dim_wrist, "dual: 두 앵커 모두 pooled dim 필요 (dinov3는 pooled: cls|clsmp)"
    print(f"[dual] anchor_main={clip_main.cache_key} (dim {dim_main}) / "
          f"anchor_wrist={clip_wrist.cache_key} (dim {dim_wrist})")
    print("인코딩/캐시 로드 중...")
    eps = ds.build(clip_main, files, verbose=False, wrist_anchor=clip_wrist)

    def stack(ids):
        return tuple(np.concatenate([eps[i][k] for i in ids])
                     for k in range(len(eps[0])))          # (Zt, Ztn, A, Zwt, Zwn)

    Zt_tr, Ztn_tr, A_tr, Zwt_tr, Zwn_tr = stack(tr_ids)
    Zt_va, Ztn_va, A_va, Zwt_va, Zwn_va = stack(val_ids)
    n_chunk, act_dim = cfg["data"]["n_chunk"], A_tr.shape[1] // cfg["data"]["n_chunk"]

    A_tr2 = A_tr.reshape(-1, act_dim)
    a_mean = A_tr2.mean(0)
    a_std = np.maximum(A_tr2.std(0), 1e-6)

    def norm_chunks(A):
        return ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std
                ).astype(np.float32)

    repr_kind = cfg["data"].get("chunk_repr", "time")
    C_tr = chunkrep.to_repr(norm_chunks(A_tr), repr_kind)
    C_va = chunkrep.to_repr(norm_chunks(A_va), repr_kind)
    Dm_tr, Dm_va = Ztn_tr - Zt_tr, Ztn_va - Zt_va           # Δz_main
    Dw_tr, Dw_va = Zwn_tr - Zwt_tr, Zwn_va - Zwt_va         # Δz_wrist
    print(f"[dual] pairs: train {len(C_tr)} / val {len(C_va)} | chunk {n_chunk}x{act_dim} "
          f"| Δz_main std {Dm_tr.std():.4f} / Δz_wrist std {Dw_tr.std():.4f}")

    wb = None
    wb_cfg = cfg.get("wandb", {})
    if wb_cfg.get("enabled") and not args.smoke:
        import wandb
        wb = wandb.init(project=wb_cfg["project"], name=wb_cfg.get("run_name"),
                        mode=wb_cfg.get("mode", "offline"), config=cfg)

    model = DualDeltaAE(act_dim, n_chunk, dim_main, dim_wrist, m_cfg["hidden"],
                        m_cfg["layers"], m_cfg["dropout"],
                        m_cfg.get("state_cond", True)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dual] DualDeltaAE params: {n_params/1e6:.2f}M "
          f"(g_main→{dim_main}, g_wrist→{dim_wrist}, h in {model.dim_cat}*2)")
    opt = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"],
                           betas=tuple(t_cfg.get("adam_betas", (0.9, 0.999))))

    loader = DataLoader(
        TensorDataset(torch.tensor(C_tr), torch.tensor(Dm_tr), torch.tensor(Dw_tr),
                      torch.tensor(Zt_tr), torch.tensor(Zwt_tr)),
        batch_size=t_cfg["batch_size"], shuffle=True, drop_last=False)
    Cv = torch.tensor(C_va, device=device)
    Dmv = torch.tensor(Dm_va, device=device)
    Dwv = torch.tensor(Dw_va, device=device)
    Zmv = torch.tensor(Zt_va, device=device)
    Zwv = torch.tensor(Zwt_va, device=device)
    epochs = 3 if args.smoke else t_cfg["epochs"]
    best_val, best_state, patience = np.inf, None, 0

    sched = None
    if t_cfg.get("scheduler") == "cosine":
        total_steps = max(1, epochs * len(loader))
        warmup = t_cfg.get("warmup_steps", 500)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            p = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    align_type = m_cfg.get("align_type", "mse_cos")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        logs, part_logs = [], []
        for c_b, dm_b, dw_b, zm_b, zw_b in loader:
            loss, parts = model.losses(c_b.to(device), dm_b.to(device),
                                       dw_b.to(device), zm_b.to(device),
                                       zw_b.to(device), w, align_type)
            opt.zero_grad(); loss.backward(); opt.step()
            if sched:
                sched.step()
            logs.append(loss.item()); part_logs.append(parts)
        model.eval()
        with torch.no_grad():
            val_loss, val_parts = model.losses(Cv, Dmv, Dwv, Zmv, Zwv, w, align_type)
        val_loss = val_loss.item()
        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
        if wb:
            train_parts = {f"train/{k}": np.mean([p[k] for p in part_logs])
                           for k in part_logs[0]}
            wb.log({"epoch": ep, "train/total": np.mean(logs),
                    "val/total": val_loss, **train_parts,
                    **{f"val/{k}": v for k, v in val_parts.items()}})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"ep {ep:3d} | train {np.mean(logs):.4f} | val {val_loss:.4f} "
                  f"({val_parts}) | patience {patience}")
        if patience >= t_cfg["early_stop_patience"]:
            print(f"early stop @ ep {ep}")
            break
    print(f"[dual] 학습 {time.time()-t0:.0f}s, best val {best_val:.4f}")
    model.load_state_dict(best_state)

    # ---- 평가 (held-out) : 디코더/왕복 R² + 스트림별 align cos ----
    model.eval()
    with torch.no_grad():
        gm = model.g_main(Cv, Zmv); gw = model.g_wrist(Cv, Zwv)
        ahat = model.decode(torch.cat([Dmv, Dwv], 1), Zmv, Zwv).cpu().numpy()
        acyc = model.decode(torch.cat([gm, gw], 1), Zmv, Zwv).cpu().numpy()
        gm_np, gw_np = gm.cpu().numpy(), gw.cpu().numpy()
    ahat = ahat.reshape(len(Cv), -1); acyc = acyc.reshape(len(Cv), -1)
    Cva = C_va.reshape(len(C_va), -1)
    dec_r2, cyc_r2 = r2(Cva, ahat), r2(Cva, acyc)

    def _cos(a, b):
        return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1)
                     * np.linalg.norm(b, axis=1) + 1e-8)))
    align_cos_m = _cos(gm_np, Dm_va)
    align_cos_w = _cos(gw_np, Dw_va)
    print(f"\n=== [dual] held-out 평가 ({len(C_va)} pairs) ===")
    print(f"디코더 h(Δz->a)  R² = {dec_r2:+.3f}   [복구]")
    print(f"cycle h(g(a))    R² = {cyc_r2:+.3f}   [복구]")
    print(f"align cos main {align_cos_m:+.3f} / wrist {align_cos_w:+.3f}   [맵핑]")

    metrics = {"decoder_r2": dec_r2, "cycle_r2": cyc_r2,
               "align_cos_main": align_cos_m, "align_cos_wrist": align_cos_w,
               "best_val_loss": float(best_val),
               "n_train_pairs": len(C_tr), "n_val_pairs": len(C_va)}
    ckpt_path = Path(os.path.expanduser(t_cfg["checkpoint"]))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg,
                "a_mean": a_mean, "a_std": a_std,
                "action_dim": act_dim, "n_chunk": n_chunk,
                "chunk_repr": repr_kind,
                "dual_stream": True, "dim_main": dim_main, "dim_wrist": dim_wrist,
                "latent_dim": model.dim_cat,
                "anchor": {"id": clip_main.id, "cache_key": clip_main.cache_key},
                "anchor_wrist": {"id": clip_wrist.id, "cache_key": clip_wrist.cache_key},
                "metrics": metrics}, ckpt_path)
    print(f"저장: {ckpt_path}")
    if args.tag:
        out = WS / "outputs" / "grid"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.tag}.json").write_text(json.dumps(
            {"tag": args.tag, "overrides": args.set, **metrics}, indent=1))
    if wb:
        wb.summary.update({"decoder_r2": dec_r2, "cycle_r2": cyc_r2,
                           "align_cos_main": align_cos_m,
                           "align_cos_wrist": align_cos_w, "n_params": n_params})
        wb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 eps, 3 epochs 코드 점검")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="config 오버라이드 (예: --set train.lr=3e-4)")
    ap.add_argument("--tag", default=None,
                    help="런 이름 — 체크포인트/지표를 grid/<tag>로 분리 저장")
    ap.add_argument("--config", default=str(CFG_PATH),
                    help="설정 yaml (기본: delta_ae.yaml)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        apply_override(cfg, kv)
    if args.tag:
        cfg["train"]["checkpoint"] = str(WS / f"checkpoints/grid/{args.tag}.pt")
        cfg["wandb"]["run_name"] = args.tag
    t_cfg, m_cfg, w = cfg["train"], cfg["model"], cfg["loss"]
    if m_cfg.get("dual_stream"):        # dual-stream 변위 AE (손목캠 추론 스트림) — 별도 경로
        return run_dual(cfg, args)      #   단일 스트림 아래 코드는 완전 무변경(비트 동형)
    rng = np.random.RandomState(t_cfg["seed"])
    torch.manual_seed(t_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 데이터 ----
    ds = get_dataset(cfg)
    files = ds.episode_files()
    if args.smoke:
        files = files[:2]
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    # 1 미만이면 비율(예: 0.2 = 20%), 이상이면 개수
    n_val = 1 if args.smoke else (max(1, round(len(files) * v)) if v < 1 else int(v))
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    print(f"episodes: train {len(tr_ids)} / val {len(val_ids)}")

    clip = get_anchor(cfg)
    m_cfg["latent_dim"] = clip.dim          # 앵커 실제 차원 반영 (CLIP joint=768 → 불변)
    print("인코딩/캐시 로드 중...")
    eps = ds.build(clip, files, verbose=False)

    def stack(ids):
        Zt = np.concatenate([eps[i][0] for i in ids])
        Ztn = np.concatenate([eps[i][1] for i in ids])
        A = np.concatenate([eps[i][2] for i in ids])
        return Zt, Ztn, A

    Zt_tr, Ztn_tr, A_tr = stack(tr_ids)
    Zt_va, Ztn_va, A_va = stack(val_ids)
    n_chunk, act_dim = cfg["data"]["n_chunk"], A_tr.shape[1] // cfg["data"]["n_chunk"]

    # 액션 정규화 (train 통계, 체크포인트에 저장)
    A_tr2 = A_tr.reshape(-1, act_dim)
    a_mean = A_tr2.mean(0)
    a_std = np.maximum(A_tr2.std(0), 1e-6)

    def norm_chunks(A):
        return ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std
                ).astype(np.float32)

    # 청크 표현 (time | dct) — 정규화 후 적용, 체크포인트에 기록 (하류 전체가 따름)
    repr_kind = cfg["data"].get("chunk_repr", "time")
    C_tr = chunkrep.to_repr(norm_chunks(A_tr), repr_kind)
    C_va = chunkrep.to_repr(norm_chunks(A_va), repr_kind)
    D_tr, D_va = Ztn_tr - Zt_tr, Ztn_va - Zt_va
    T_tr, T_va = D_tr, D_va
    Ddec_tr, Ddec_va = D_tr, D_va
    print(f"pairs: train {len(C_tr)} / val {len(C_va)} | chunk {n_chunk}x{act_dim}")

    # ---- 언어정렬(C8/HY03): 모션 문장 (align_mode direct/hybrid 전용) ----
    align_mode = m_cfg.get("align_mode", "dz")
    sent_emb_t = ids_va_t = None
    if align_mode != "dz":
        from data.motion_lang import MotionSentences
        ms = MotionSentences(version=cfg["data"].get("motion_vocab", "v1"))
        ids_tr = ms.assign(A_tr.reshape(-1, n_chunk, act_dim))
        ids_va = ms.assign(A_va.reshape(-1, n_chunk, act_dim))
        sent_emb = ms.embed_all(clip)          # (n_sent, dim_text) — 앵커 텍스트 인코더
        print(f"C8 align_mode={align_mode}: 고유 문장 {len(ms.sentences)}, "
              f"train 사용 {len(np.unique(ids_tr))}종")
        sent_emb_t = torch.tensor(sent_emb, device=device)
        ids_va_t = torch.tensor(ids_va, device=device)
        ids_tr_t = torch.tensor(ids_tr)
    else:
        ids_tr_t = torch.zeros(len(C_tr), dtype=torch.long)   # dz: 미사용 placeholder

    # ---- wandb ----
    wb = None
    wb_cfg = cfg.get("wandb", {})
    if wb_cfg.get("enabled") and not args.smoke:
        import wandb
        wb = wandb.init(project=wb_cfg["project"], name=wb_cfg.get("run_name"),
                        mode=wb_cfg.get("mode", "offline"), config=cfg)

    # ---- 모델/학습 ----
    model = DeltaAE(act_dim, n_chunk, m_cfg["latent_dim"], m_cfg["hidden"],
                    m_cfg["layers"], m_cfg["dropout"],
                    m_cfg.get("state_cond", True),
                    m_cfg.get("decoder_state_cond"),
                    m_cfg.get("encoder_state_cond"),
                    align_mode=align_mode,
                    contrast_w=float(w.get("contrast", 0.0)),
                    contrast_loss=m_cfg.get("contrast_loss", "infonce"),
                    contrast_head=m_cfg.get("contrast_head", False),
                    sigmoid_bias0=m_cfg.get("sigmoid_bias0", -5.5),
                    align_block=m_cfg.get("align_block"),
                    h_mode=m_cfg.get("h_mode", "mlp"),
                    h_flow_steps=m_cfg.get("h_flow_steps", 5)).to(device)
    if model.h_mode == "flow":                    # S2: h-flow x0 스케일 = 액션청크 타깃 std
        model.h.x0_std.fill_(float(np.asarray(C_tr).reshape(len(C_tr), -1).std()))
        print(f"h-flow decoder: steps={model.h.steps}, x0_std={model.h.x0_std.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeltaAE params: {n_params/1e6:.2f}M (encoder cnn/{m_cfg['hidden']}x"
          f"{m_cfg['layers']}, decoder {model.h_mode}/{m_cfg['hidden']}x{m_cfg['layers']})")
    opt = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"],
                           betas=tuple(t_cfg.get("adam_betas", (0.9, 0.999))))

    # DataLoader: shuffle=True -> 매 epoch 전체 시점을 랜덤 순서로 배치 구성
    loader = DataLoader(
        TensorDataset(torch.tensor(C_tr), torch.tensor(T_tr),
                      torch.tensor(Zt_tr), ids_tr_t),
        batch_size=t_cfg["batch_size"], shuffle=True, drop_last=False)
    Cv = torch.tensor(C_va, device=device)
    Dv = torch.tensor(T_va, device=device)
    Zv = torch.tensor(Zt_va, device=device)
    epochs = 3 if args.smoke else t_cfg["epochs"]
    best_val, best_state, patience = np.inf, None, 0

    # cosine + warmup 스케줄러 (VITA Table 10 관례)
    sched = None
    if t_cfg.get("scheduler") == "cosine":
        total_steps = max(1, epochs * len(loader))
        warmup = t_cfg.get("warmup_steps", 500)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            p = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    align_type = m_cfg.get("align_type", "mse_cos")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        logs, part_logs = [], []
        for chunk_b, delta_b, zt_b, ids_b in loader:
            kw = {}
            if align_mode != "dz":
                ids_b = ids_b.to(device)
                kw = {"text_emb": sent_emb_t[ids_b], "sent_ids": ids_b}
            loss, parts = model.losses(chunk_b.to(device), delta_b.to(device),
                                       w, zt_b.to(device), align_type, **kw)
            opt.zero_grad(); loss.backward(); opt.step()
            if sched:
                sched.step()
            logs.append(loss.item()); part_logs.append(parts)
        model.eval()
        with torch.no_grad():
            kw_v = ({"text_emb": sent_emb_t[ids_va_t], "sent_ids": ids_va_t}
                    if align_mode != "dz" else {})
            val_loss, val_parts = model.losses(Cv, Dv, w, Zv, align_type, **kw_v)
        val_loss = val_loss.item()
        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
        if wb:
            train_parts = {f"train/{k}": np.mean([p[k] for p in part_logs])
                           for k in part_logs[0]}
            wb.log({"epoch": ep, "train/total": np.mean(logs),
                    "val/total": val_loss,
                    **train_parts,
                    **{f"val/{k}": v for k, v in val_parts.items()}})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"ep {ep:3d} | train {np.mean(logs):.4f} | val {val_loss:.4f} "
                  f"({val_parts}) | patience {patience}")
        if patience >= t_cfg["early_stop_patience"]:
            print(f"early stop @ ep {ep}")
            break
    print(f"학습 {time.time()-t0:.0f}s, best val {best_val:.4f}")
    model.load_state_dict(best_state)

    # ---- 평가 (held-out 에피소드) ----
    model.eval()
    with torch.no_grad():
        ghat = model.g(Cv, Zv).cpu().numpy()
        ahat = model.h(Dv, Zv).cpu().numpy().reshape(len(Cv), -1)
        acyc = model.h(model.g(Cv, Zv), Zv).cpu().numpy().reshape(len(Cv), -1)
    Cva = C_va.reshape(len(C_va), -1)
    dec_r2, cyc_r2 = r2(Cva, ahat), r2(Cva, acyc)
    if getattr(model, "h_mode", "mlp") == "flow":   # 생성 디코더: 단일샘플 R²는 낮음(분포서 모드 샘플).
        with torch.no_grad():                        #   K샘플 평균-R²가 MLP급이면 수렴·조건분산(다봉), 낮으면 미수렴.
            K = 32
            gz = model.g(Cv, Zv)
            mrec = np.mean([model.h(Dv, Zv).cpu().numpy().reshape(len(Cv), -1)
                            for _ in range(K)], axis=0)
            mcyc = np.mean([model.h(gz, Zv).cpu().numpy().reshape(len(Cv), -1)
                            for _ in range(K)], axis=0)
        print(f"[h-flow K={K} 평균] recon 평균-R² = {r2(Cva, mrec):+.3f} / "
              f"cycle 평균-R² = {r2(Cva, mcyc):+.3f} (단일 {dec_r2:+.3f}/{cyc_r2:+.3f})")
    # 맵핑 정렬도: g(a)와 실제 Δz의 평균 cosine
    align_cos = float(np.mean(
        (ghat * T_va).sum(1)
        / (np.linalg.norm(ghat, axis=1) * np.linalg.norm(T_va, axis=1) + 1e-8)))
    t1a, t5a = retrieval(ghat, T_va)
    t1z, t5z = retrieval(T_va, ghat)
    chance1, chance5 = 100 / len(C_va), 500 / len(C_va)
    print(f"\n=== held-out 평가 ({len(C_va)} pairs) ===")
    print(f"디코더 h(Δz->a)  R² = {dec_r2:+.3f}   [복구]")
    print(f"cycle h(g(a))    R² = {cyc_r2:+.3f}   [복구]")
    print(f"align cos(g(a), Δz) = {align_cos:+.3f}   [맵핑]")
    print(f"검색 액션->Δz  top-1 {t1a:.1f}% top-5 {t5a:.1f}% (우연 {chance1:.1f}/{chance5:.1f}%) [맵핑]")
    print(f"검색 Δz->액션  top-1 {t1z:.1f}% top-5 {t5z:.1f}%")

    # ---- 저장 ----
    metrics = {"decoder_r2": dec_r2, "cycle_r2": cyc_r2,
               "align_cos": align_cos, "best_val_loss": float(best_val),
               "retrieval_a2z": [t1a, t5a], "retrieval_z2a": [t1z, t5z],
               "n_train_pairs": len(C_tr), "n_val_pairs": len(C_va)}
    ckpt_path = Path(os.path.expanduser(t_cfg["checkpoint"]))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg,
                "a_mean": a_mean, "a_std": a_std,
                "action_dim": act_dim, "n_chunk": n_chunk,
                "chunk_repr": repr_kind, "latent_dim": clip.dim,
                "anchor": {"id": clip.id, "cache_key": clip.cache_key},
                "metrics": metrics}, ckpt_path)
    print(f"저장: {ckpt_path}")
    if args.tag:  # 그리드서치용 지표 json
        out = WS / "outputs" / "grid"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.tag}.json").write_text(json.dumps(
            {"tag": args.tag, "overrides": args.set, **metrics}, indent=1))
    if wb:
        wb.summary.update({"decoder_r2": dec_r2, "cycle_r2": cyc_r2,
                           "retrieval_a2z_top1": t1a, "retrieval_z2a_top1": t1z,
                           "n_params": n_params})
        wb.finish()


if __name__ == "__main__":
    main()
