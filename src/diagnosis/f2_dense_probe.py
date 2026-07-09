"""F2 — dense 디코더빌리티 프로브 (DESIGN §F2, 전체 dense 가설 go/no-go).

인코더별 표현 → GT action chunk 를 **선형(RidgeCV)·얕은(MLP) 프로브**로 회귀,
held-out R²·MAE 측정. 오프라인 (sim·폐루프·flow 학습 없음). 로컬 실행 가능.

입력(§F2 "Δ(표현)→action"을 phase1 디코더 h(Δz,z_t)와 정합):
  - 기본 --state-cond: [Δ표현, z_t표현]  (phase1 h와 동일한 상태조건부 — CLIP이 phase1
    dec R²≈0.68 근처로 복원되면 프로브 유효성 sanity 통과)
  - --no-state-cond: Δ표현만 (더 엄격하지만 상태 미결정성으로 저조)

주의(§0.3 불변식): 오프라인 디코더빌리티 ≠ 폐루프 우세 (proprio·DINOv2 반례). F2 통과는
F3/F4 진행의 필요조건이지 충분조건 아님. 백본/설정 선정은 반드시 폐루프로.

팔(§F2): CLIP-pooled(기준) / SigLIP2-pooled / DINOv2-cls / DINOv2-clsmp(CLS⊕patch-mean=
dense-informed) / 융합(SigLIP2 ⊕ DINOv2-clsmp). RADIO-spatial은 다운로드 필요 → 후속.
(진짜 dense attention-pool·patch 필드는 F3/F4; 여기선 patch-mean을 dense 대리로 사용.)

  python src/diagnosis/f2_dense_probe.py --episodes 60 --mlp --suite libero_spatial
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

from core.anchor import get_anchor
from data.libero import LiberoDataset

WS = Path(__file__).resolve().parents[2]

# (라벨, [앵커 스펙...]) — 여러 스펙이면 특징축 concat(융합)
ARMS = [
    ("clip-pooled",      [{"name": "clip"}]),
    ("siglip2-pooled",   [{"name": "siglip2"}]),
    ("dinov2-cls",       [{"name": "dinov2", "pooled": "cls"}]),
    ("dinov2-clsmp",     [{"name": "dinov2", "pooled": "clsmp"}]),
    ("fusion-sig+dino",  [{"name": "siglip2"}, {"name": "dinov2", "pooled": "clsmp"}]),
]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def build_feats(ds, files, specs):
    """specs 앵커들로 (Δ표현, z_t표현, action chunk). 융합이면 특징축 concat.
    (CLIP 섹션은 get_anchor→ClipAnchor가 load_config()로 config.yaml에서 읽음)"""
    dz, zt, A_ref = [], [], None
    for spec in specs:
        anc = get_anchor({"anchor": spec})
        eps = ds.build(anc, files, verbose=False)
        Zt = np.concatenate([e[0] for e in eps])
        Ztn = np.concatenate([e[1] for e in eps])
        A_ref = np.concatenate([e[2] for e in eps])   # 앵커 공통 (동일 starts)
        dz.append((Ztn - Zt).astype(np.float32))
        zt.append(Zt.astype(np.float32))
        del anc
        torch.cuda.empty_cache()
    return np.concatenate(dz, axis=1), np.concatenate(zt, axis=1), A_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase1_libero.yaml"))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp", action="store_true", help="얕은 MLP 프로브도 실행")
    ap.add_argument("--state-cond", dest="state_cond", action="store_true", default=True)
    ap.add_argument("--no-state-cond", dest="state_cond", action="store_false")
    ap.add_argument("--suite", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.suite:
        cfg["data"]["root"] = str(Path(cfg["data"]["root"]).parent / args.suite)
    ds = LiberoDataset(cfg)
    files = ds.episode_files()
    rng = np.random.RandomState(args.seed)
    files = [files[i] for i in rng.permutation(len(files))[:args.episodes]]
    n_val = max(1, round(len(files) * args.val_frac))
    val_files, tr_files = files[:n_val], files[n_val:]
    n_chunk = cfg["data"]["n_chunk"]
    print(f"F2 probe | suite={Path(cfg['data']['root']).name} | ep tr {len(tr_files)} / "
          f"val {len(val_files)} | state_cond={args.state_cond} | seed {args.seed}")

    results, a_mean, a_std = {}, None, None
    for label, specs in ARMS:
        t0 = time.time()
        try:
            DZ_tr, ZT_tr, Atr = build_feats(ds, tr_files, specs)
            DZ_va, ZT_va, Ava = build_feats(ds, val_files, specs)
        except Exception as e:
            print(f"  [{label}] SKIP ({type(e).__name__}: {str(e)[:90]})")
            continue
        Xtr = np.concatenate([DZ_tr, ZT_tr], 1) if args.state_cond else DZ_tr
        Xva = np.concatenate([DZ_va, ZT_va], 1) if args.state_cond else DZ_va
        act_dim = Atr.shape[1] // n_chunk
        if a_mean is None:
            a2 = Atr.reshape(-1, act_dim)
            a_mean, a_std = a2.mean(0), np.maximum(a2.std(0), 1e-6)

        def norm_y(A):
            return ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std
                    ).reshape(len(A), -1)
        Ytr, Yva = norm_y(Atr), norm_y(Ava)
        mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)   # z-score (train 통계)
        Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd

        rid = RidgeCV(alphas=ALPHAS).fit(Xtr_n, Ytr)
        pv = rid.predict(Xva_n)
        r = {"dim": int(Xtr.shape[1]), "n_tr": int(len(Xtr)), "n_va": int(len(Xva)),
             "ridge_r2": float(r2_score(Yva, pv)),
             "ridge_mae": float(mean_absolute_error(Yva, pv)),
             "ridge_alpha": float(rid.alpha_)}
        if args.mlp:
            mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=400,
                               early_stopping=True, random_state=args.seed).fit(Xtr_n, Ytr)
            r["mlp_r2"] = float(r2_score(Yva, mlp.predict(Xva_n)))
        r["sec"] = round(time.time() - t0, 1)
        results[label] = r
        extra = f" mlp_r2 {r['mlp_r2']:+.3f}" if args.mlp else ""
        print(f"  [{label:16s}] dim {r['dim']:5d} | ridgeCV_r2 {r['ridge_r2']:+.3f}"
              f" (α{r['ridge_alpha']:g}){extra} | {r['sec']}s")

    base = results.get("clip-pooled", {}).get("ridge_r2")
    base_mlp = results.get("clip-pooled", {}).get("mlp_r2")
    print("\n=== F2 요약 (표현→GT action, held-out R²; CLIP-pooled 기준) ===")
    print("  [sanity] CLIP state-cond R²가 phase1 dec R²≈0.68 근처면 프로브 유효")
    for label, r in results.items():
        dr = f" (vs clip {r['ridge_r2'] - base:+.3f})" if base is not None else ""
        dm = (f" | mlp {r.get('mlp_r2', float('nan')):+.3f}"
              f" (vs clip {r.get('mlp_r2', 0) - base_mlp:+.3f})"
              if base_mlp is not None and "mlp_r2" in r else "")
        print(f"  {label:16s} ridgeCV {r['ridge_r2']:+.3f}{dr}{dm}")

    out = WS / "outputs" / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "f2_dense_probe.json").write_text(json.dumps(
        {"probe": "f2_dense_decodability", "suite": Path(cfg["data"]["root"]).name,
         "episodes": args.episodes, "state_cond": args.state_cond, "seed": args.seed,
         "arms": results, "clip_baseline_ridge_r2": base}, indent=1))
    print(f"\n저장: {out / 'f2_dense_probe.json'}")


if __name__ == "__main__":
    main()
