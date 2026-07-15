"""Phase 2 — 잠재 정책 f 학습 (인코더·디코더 동결 = Stage A).

  샘플: (z_prev, z_cur, z_next, A_past, A_fut) 연속 윈도우 삼중쌍
  f 입력토큰: [z_prev, z_cur, g(A_past, z_prev)]  → ζ̂
  평가: 관절 MAE(°), 잠재 cos(vs g타깃/vs Δz타깃), 디코딩 액션 R², 평균붕괴 진단

사용 (clipx env):
  python src/training/train_phase2.py            # 본 학습 (configs/phase2.yaml)
  python src/training/train_phase2.py --smoke    # 코드 점검 (2 eps)
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
from models.networks import DeltaAE
from models.policy import FlowPolicy, build_policy_from_cfg, policy_losses

WS = Path(__file__).resolve().parents[2]
CFG_PATH = WS / "configs" / "phase2.yaml"


def r2(y, yhat):
    dev = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / (dev + 1e-12))


def apply_override(cfg, kv):
    key, val = kv.split("=", 1)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = yaml.safe_load(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="학습 에피소드 수 제한 (F3 공정 subset — 고정시드로 arm 간 동일 표본)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default=str(CFG_PATH))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for kv in args.set:
        apply_override(cfg, kv)
    if args.tag:
        cfg["train"]["checkpoint"] = str(WS / f"checkpoints/grid/{args.tag}.pt")
        cfg["wandb"]["run_name"] = args.tag
    t_cfg, m_cfg, w = cfg["train"], cfg["module"], cfg["loss"]
    rng = np.random.RandomState(t_cfg["seed"])
    torch.manual_seed(t_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- phase1 동결 모델 (g/h + 정규화 통계) ----
    ck = torch.load(os.path.expanduser(cfg["phase1_ckpt"]),
                    map_location="cpu", weights_only=False)
    p1 = ck["config"]
    n_chunk, act_dim = ck["n_chunk"], ck["action_dim"]
    a_mean, a_std = ck["a_mean"], ck["a_std"]
    repr_kind = ck.get("chunk_repr", "time")      # phase1이 정한 청크 표현을 따름
    ae = DeltaAE(act_dim, n_chunk, p1["model"]["latent_dim"],
                 p1["model"]["hidden"], p1["model"]["layers"],
                 p1["model"]["dropout"],
                 p1["model"].get("state_cond", True),
                 p1["model"].get("decoder_state_cond"),
                 p1["model"].get("encoder_state_cond")).to(device)
    ae.load_state_dict(ck["state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # ---- 데이터 (삼중쌍) ----
    ds = get_dataset(cfg)
    files = ds.episode_files()
    if args.smoke:
        files = files[:2]
    elif args.max_episodes:                       # F3 공정 subset (고정시드=arm 간 동일 표본)
        sub = np.random.RandomState(0).permutation(len(files))[:args.max_episodes]
        files = [files[i] for i in sub]
    perm = rng.permutation(len(files))
    v = cfg["data"]["val_episodes"]
    # 1 미만이면 비율(예: 0.2 = 20%), 이상이면 개수
    n_val = 1 if args.smoke else (max(1, round(len(files) * v)) if v < 1 else int(v))
    val_ids, tr_ids = perm[:n_val], perm[n_val:]
    clip = get_anchor(cfg)          # 앵커 config 반영 (무-anchor면 ClipAnchor=ClipWrapper와 동일)

    # ---- S1b 비대칭 역할분리 융합 (cond_anchor 있을 때만; 없으면 아래 경로 전부 no-op = 비트 동형) ----
    #  설계: 정책 flow 조건 토큰(z_prev/z_cur/lang/wrist)=SigLIP2-alone(의미·언어 경로) /
    #        DeltaAE 경로(aemb=g, ζ 타깃=g, 디코딩 h, world-model)=융합 concat 전체(기하).
    #  fused = dualconcat[L2norm(SigLIP2); L2norm(DINOv3)] 이므로 SigLIP2-alone(normalize=true)
    #    == fused[:, :cond_dim] 로 bit-exact (동일 SigLIP2 모델·동일 L2정규화). → 별도 SigLIP2
    #    캐시 재로딩 불필요: 융합 z 를 슬라이스하면 분리 스트림과 완전 동일(데이터 파이프라인 무변경).
    #  주의: 기존 libero_emb_large256 캐시는 normalize=false 라 서브블록(norm=true)과 불일치 →
    #        슬라이스가 cond_anchor(normalize=true) 규격에 맞는 유일한 정확 소스.
    cond_dim = None
    cond_cfg = cfg.get("cond_anchor")
    if cond_cfg:
        # dualconcat 은 이미 _sig 서브인코더를 보유 → 그 dim 재사용(추가 모델 로드 없음).
        # 그 외 융합앵커라면 cond_anchor 를 직접 구성해 dim 조회.
        cond_dim = (clip._sig.dim if hasattr(clip, "_sig")
                    else get_anchor({"anchor": cond_cfg}).dim)
        assert 0 < cond_dim < clip.dim, \
            f"cond_dim {cond_dim} vs fused dim {clip.dim}: SigLIP2 서브블록 폭(<융합폭)이어야 함"
        print(f"S1b 역할분리: 조건 토큰=SigLIP2[0:{cond_dim}] / g·h·ζ·wm=융합 concat[{clip.dim}]")

    # ---- F3 관측 융합 앵커 (module.obs 있을 때만; 없으면 기존 no-obs 경로와 완전 동일) ----
    obs_cfg = m_cfg.get("obs")
    obs_anchors, enc_dims, enc_names = None, {}, []
    if obs_cfg:
        obs_anchors = []
        for enc in obs_cfg["encoders"]:
            anc = get_anchor({"anchor": enc})
            if enc["name"] == "siglip2":
                anc.save_tokens = True                # siglip2: 패치 토큰 반환 활성화
            obs_anchors.append((enc["name"], anc, enc.get("camera", "agentview_rgb")))
            enc_dims[enc["name"]] = anc.patch_dim     # dinov2=1024 / siglip2=1152
            enc_names.append(enc["name"])

    # ---- C1/F4 fine 채널 (module.f4.enable 있을 때만; 없으면 아래 경로 전부 no-op = 비트 동형) ----
    f4_cfg = m_cfg.get("f4")
    f4_on = bool(f4_cfg and f4_cfg.get("enable"))
    f4_anchor = None
    if f4_on:
        assert not obs_cfg, "module.f4 와 module.obs 는 동시 사용 불가 (C1 = 단일 dense 채널)"
        assert m_cfg.get("name") == "flow", "F4 fine flow 는 flow 정책 전제"
        de = f4_cfg["dense"]                          # dense ΔF 앵커 (siglip2-large-256 = C1 기질)
        f4_dense_anc = get_anchor({"anchor": de})
        if de["name"] == "siglip2":
            f4_dense_anc.save_tokens = True
        f4_anchor = (f4_dense_anc, de.get("camera", "agentview_rgb"))

    print("삼중쌍 구성 중 (임베딩 캐시 재사용)...")
    eps = ds.build_policy_samples(clip, files, stride=cfg["data"].get("stride", 2),
                                  obs_anchors=obs_anchors, f4_anchor=f4_anchor)

    def stack(ids):
        return tuple(np.concatenate([eps[i][k] for i in ids])
                     for k in range(len(eps[0])))

    Zp_tr, Zc_tr, Zn_tr, Ap_tr, Af_tr, *Wx_tr = stack(tr_ids)
    Zp_va, Zc_va, Zn_va, Ap_va, Af_va, *Wx_va = stack(val_ids)

    # 손목캠 토큰: 로더가 6번째 배열(z_wrist)을 준 경우에만 사용 가능.
    # 6번째 이후는 F3 관측 dense 배열(인코더 순서). n_wrist로 잘라 분리한다.
    use_wrist = m_cfg.get("wrist_token", False)
    if use_wrist and not ds.wrist_camera:
        raise ValueError("module.wrist_token=true지만 data.wrist_camera 미설정")
    n_wrist = 1 if ds.wrist_camera else 0
    W_tr = Wx_tr[0] if use_wrist else None
    W_va = Wx_va[0] if use_wrist else None
    # 인코더별 dense [N,P,d]. obs=F3 관측 인코더들 / f4=단일 patch ΔF (맨 뒤 1개).
    Dobs_tr = Wx_tr[n_wrist:] if (obs_cfg or f4_on) else []
    Dobs_va = Wx_va[n_wrist:] if (obs_cfg or f4_on) else []

    # ---- Exp2 both-aug: variant 뱅크 처리 (data.augment on일 때만) ----
    # build_policy_samples가 (N, M, D) 뱅크를 주면: val/eval은 항상 variant0(클린),
    # train은 forward_train에서 샘플별 랜덤 variant를 뽑는다(zp/zc/zn 일관).
    # augment off면 모든 배열이 2D → 아래 _v0/pick 모두 no-op (기존과 완전 동일).
    def _v0(x):
        return x[:, 0] if x is not None and getattr(x, "ndim", 0) == 3 else x
    Zp_va, Zc_va, Zn_va = _v0(Zp_va), _v0(Zc_va), _v0(Zn_va)
    W_va = _v0(W_va)
    Zc_tr0 = _v0(Zc_tr)                               # 클린 통계(x0_std)용 variant0

    # 언어 토큰 (멀티태스크 조건화): 에피소드별 지시문 임베딩을 샘플 수만큼 복제
    use_lang = m_cfg.get("lang_token", False)
    if f4_on:
        assert use_lang, "module.f4 는 쿼리 초기화용 텍스트 임베딩 필요 → lang_token=true 전제"
    if use_lang:
        lang_per_ep = [ds.instruction_embedding(clip, files[i])
                       for i in range(len(files))]

        def stack_lang(ids):
            return np.concatenate([
                np.repeat(lang_per_ep[i][None], len(eps[i][0]), axis=0)
                for i in ids]).astype(np.float32)
        L_tr, L_va = stack_lang(tr_ids), stack_lang(val_ids)

    def norm(A):
        a = ((A.reshape(len(A), n_chunk, act_dim) - a_mean) / a_std
             ).astype(np.float32)
        return chunkrep.to_repr(a, repr_kind)

    Cp_tr, Cf_tr = norm(Ap_tr), norm(Af_tr)
    Cp_va, Cf_va = norm(Ap_va), norm(Af_va)
    print(f"samples: train {len(Cf_tr)} / val {len(Cf_va)} | chunk {n_chunk}x{act_dim}")

    # 과거 청크 임베딩: 학습 중 노이즈 주입을 위해 val만 사전 계산
    past_noise = float(t_cfg.get("past_noise", 0.0))
    with torch.no_grad():
        def embed_past(Cp, Zp):
            out = []
            for i in range(0, len(Cp), 4096):
                out.append(ae.g(torch.tensor(Cp[i:i+4096], device=device),
                                torch.tensor(Zp[i:i+4096], device=device)).cpu())
            return torch.cat(out)
        Ae_va = embed_past(Cp_va, Zp_va)

    # ---- wandb ----
    wb = None
    wb_cfg = cfg.get("wandb", {})
    if wb_cfg.get("enabled") and not args.smoke:
        import wandb
        wb = wandb.init(project=wb_cfg["project"], name=wb_cfg.get("run_name"),
                        mode=wb_cfg.get("mode", "online"), config=cfg)

    # ---- F3 관측 융합 모듈 (obs일 때만; K개 관측 토큰을 정책 입력열 끝에 추가) ----
    obs_fusion, K = None, 0
    if obs_cfg:
        from models.obs_fusion import ObsFusion
        obs_fusion = ObsFusion(enc_dims, d_attn=obs_cfg.get("d_attn", 768),
                               n_query=obs_cfg.get("n_query", 8),
                               out_dim=p1["model"]["latent_dim"],
                               pool=obs_cfg.get("pool", "attn"),
                               unshuffle=obs_cfg.get("unshuffle", 1)).to(device)
        K = obs_fusion.n_query                        # attn: n_query / mean: 1

    # ---- 정책 모델 ----
    n_tokens = 3 + int(use_lang) + int(use_wrist) + K
    model = build_policy_from_cfg(m_cfg, n_tokens=n_tokens,
                                  latent_dim=p1["model"]["latent_dim"]).to(device)
    is_flow = isinstance(model, FlowPolicy)

    # ---- C1/F4 fine 채널 (정책 뒤에 구성 → 정책 파라미터 RNG 소비 불변 = bit-identity) ----
    f4 = None
    if f4_on:
        from models.f4 import build_f4_from_cfg
        n_base = 3 + int(use_lang) + int(use_wrist)   # f4 flow 조건 = base 토큰(관측/미래 제외)
        dense_dim = Dobs_tr[0].shape[-1]              # ΔF 마지막축 (siglip2-large-256=1024)
        n_patch = Dobs_tr[0].shape[-2]                # 256 (16×16)
        f4 = build_f4_from_cfg(f4_cfg, dense_dim=dense_dim, text_dim=L_tr.shape[1],
                               latent_dim=p1["model"]["latent_dim"], n_base_tokens=n_base,
                               action_dim=act_dim, n_chunk=n_chunk, n_patch=n_patch).to(device)
        print(f"f4: K={f4.K}, bottleneck={f4.bneck}, ζ_f dim={f4.zf_dim}, "
              f"dense ΔF={n_patch}×{dense_dim}, query_init=text, gate=tanh(α={float(f4.alpha):.1f})")
    if is_flow:                                   # x0 스케일 = 잠재 타깃 분포
        with torch.no_grad():
            lt = ae.g(torch.tensor(Cf_tr[:4096], device=device),
                      torch.tensor(Zc_tr0[:4096], device=device))
        model.x0_std.fill_(lt.std().item())
        print(f"flow: source={model.source}, steps={model.steps}, "
              f"x0_std={model.x0_std.item():.4f}")
    if f4_on:                                     # ζ_f flow source std = 게이트 전 병목 스케일
        with torch.no_grad():
            dF0 = torch.tensor(Dobs_tr[0][:2048], device=device)
            l0 = torch.tensor(L_tr[:2048], device=device)
            zf_scale = f4.source_std(dF0, l0)     # 게이트 전 병목 std (fine_mode 무관)
        f4.zf_std.fill_(max(zf_scale, 1e-2))
        print(f"f4: ζ_f flow source std={f4.zf_std.item():.4f}, "
              f"params {sum(p.numel() for p in f4.parameters())/1e6:.2f}M")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"policy[{m_cfg['name']}] params: {n_params/1e6:.2f}M "
          f"(d{m_cfg['d_model']}/L{m_cfg['layers']}/H{m_cfg.get('heads', 8)})")
    opt = torch.optim.Adam(
        list(model.parameters())
        + (list(obs_fusion.parameters()) if obs_cfg else [])
        + (list(f4.parameters()) if f4_on else []),
        lr=t_cfg["lr"],
        betas=tuple(t_cfg.get("adam_betas", (0.9, 0.999))))

    L_tr_t = torch.tensor(L_tr) if use_lang else torch.zeros(len(Cp_tr), 0)
    W_tr_t = torch.tensor(W_tr) if use_wrist else torch.zeros(len(Cp_tr), 0)
    Dobs_tr_t = [torch.tensor(d) for d in Dobs_tr]     # F3: 인코더별 dense (없으면 [])
    loader = DataLoader(
        TensorDataset(torch.tensor(Zp_tr), torch.tensor(Zc_tr),
                      torch.tensor(Zn_tr), torch.tensor(Cp_tr),
                      torch.tensor(Cf_tr), L_tr_t, W_tr_t, *Dobs_tr_t),
        batch_size=t_cfg["batch_size"], shuffle=True)
    val_t = [torch.tensor(x, device=device) for x in (Zp_va, Zc_va, Zn_va)] \
        + [Ae_va.to(device), torch.tensor(Cf_va, device=device)] \
        + [torch.tensor(L_va, device=device) if use_lang
           else torch.zeros(len(Cf_va), 0, device=device)] \
        + [torch.tensor(W_va, device=device) if use_wrist
           else torch.zeros(len(Cf_va), 0, device=device)] \
        + [torch.tensor(d, device=device) for d in Dobs_va]
    epochs = 3 if args.smoke else t_cfg["epochs"]
    best_val, best_state, best_f4, patience = np.inf, None, None, 0

    sched = None
    if t_cfg.get("scheduler") == "cosine":
        total_steps = max(1, epochs * len(loader))
        warmup = t_cfg.get("warmup_steps", 500)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    lam_fcfm = w.get("f_cfm", 0.5)                    # C1/F4 손실 가중 (f4 off면 미사용)
    lam_consist = w.get("consist", 0.1)

    def forward(zp, zc, zn, aemb, cf, lang, wr, *dobs):
        zdim = zp.shape[-1]                           # DeltaAE 경로 폭 (dualconcat 융합=2048; 기존=앵커 폭)
        # ── S1b 역할분리: 조건 토큰 z_prev/z_cur/(wrist)는 SigLIP2 서브블록[0:cond_dim]만 사용
        #    (의미·언어 경로). g/h/ζ/wm 은 아래에서 항상 융합 z 전체(zp/zc/zn 원본) 사용.
        #    cond_dim=None(cond_anchor 미설정)이면 zp_c=zp … → 아래 전부 no-op(비트 동형).
        zp_c, zc_c, wr_c = zp, zc, wr
        if cond_dim:                                  # fused[:, :cond_dim] == SigLIP2-alone(norm=true)
            zp_c, zc_c = zp[:, :cond_dim], zc[:, :cond_dim]
            if use_wrist:
                wr_c = wr[:, :cond_dim]

        def _pad(t):                                  # 좁은 조건 토큰을 z 폭의 SigLIP2 블록[0:d]에 zero-pad
            return t if t.shape[-1] == zdim else \
                torch.nn.functional.pad(t, (0, zdim - t.shape[-1]))   # (언어정렬=SigLIP2 블록만, 학습 0)
        base = [_pad(zp_c), _pad(zc_c), aemb] + ([_pad(lang)] if use_lang else []) \
            + ([_pad(wr_c)] if use_wrist else [])     # f4 flow 조건 = base 토큰 (미래 무접근)
        toks = list(base)
        if obs_cfg:                                   # F3: 관측 토큰 K개를 열 끝에 추가
            obs_tok = obs_fusion({n: d for n, d in zip(enc_names, dobs)})  # (B,K,768)
            toks = toks + [obs_tok[:, k] for k in range(K)]
        toks = torch.stack(toks, dim=1)               # (B, 3~5+K, 768)
        if is_flow:
            # lat 자리 = CFM 손실, act = FLD(ODE 샘플 디코딩). val은 고정시드로 결정화
            gen = None
            if not model.training:
                gen = torch.Generator(device=device); gen.manual_seed(0)
            with torch.no_grad():
                lat_target = ae.g(cf, zc)
            zeta, l_fm = model.fm_and_sample(toks, lat_target, generator=gen)   # ζ̂_g (정책 무변경)
            ahat = ae.h(zeta, zc)                      # frozen 디코딩 (pooled 채널)
            cos = torch.nn.functional.cosine_similarity
            l_wm = 0.5 * (1 - cos(zeta, zn - zc, dim=1)).mean()
            parts = {}
            if f4_on:                                 # ζ_f: 미래 ΔF에서 인코딩(타깃), flow로 생성
                zeta_f_tgt = f4.encode(dobs[0], lang)            # (B, K*bneck), 초기 0(α=0)
                zeta_f, l_f_fm = f4.flow_fm_and_sample(
                    torch.cat(base, dim=1), zeta_f_tgt, generator=gen)
                ahat = ahat + f4.fine_action(zeta, zeta_f, zc)  # gated 잔차(β=0 → 초기 0)
                l_consist = f4.consistency(zeta_f_tgt, zn - zc)
                parts.update(f_cfm=l_f_fm.item(), consist=l_consist.item())
            l_act = torch.nn.functional.l1_loss(ahat, cf)
            total = w["lat"] * l_fm + w["act"] * l_act + w["wm"] * l_wm
            if f4_on:
                total = total + lam_fcfm * l_f_fm + lam_consist * l_consist
            parts.update(lat=l_fm.item(), act=l_act.item(), wm=l_wm.item())
            return total, parts
        zeta = model(toks)
        return policy_losses(zeta, cf, zc, zn, ae, w)

    def forward_train(zp, zc, zn, cp, cf, lang, wr, *dobs):
        if zp.dim() == 3:                             # Exp2 aug 뱅크: 샘플별 랜덤 variant
            B, M = zp.shape[0], zp.shape[1]           # (zp/zc/zn 동일 m — 일관 뷰)
            m = torch.randint(0, M, (B, 1, 1), device=zp.device)
            zp = zp.gather(1, m.expand(-1, 1, zp.shape[2])).squeeze(1)
            zc = zc.gather(1, m.expand(-1, 1, zc.shape[2])).squeeze(1)
            zn = zn.gather(1, m.expand(-1, 1, zn.shape[2])).squeeze(1)
        if wr.dim() == 3:                             # 손목캠 aug 뱅크: 독립 variant
            Bw, Mw = wr.shape[0], wr.shape[1]
            mw = torch.randint(0, Mw, (Bw, 1, 1), device=wr.device)
            wr = wr.gather(1, mw.expand(-1, 1, wr.shape[2])).squeeze(1)
        if past_noise > 0:                            # 폐루프 오차 누적 모사
            cp = cp + torch.randn_like(cp) * past_noise
        with torch.no_grad():
            aemb = ae.g(cp, zp)
        return forward(zp, zc, zn, aemb, cf, lang, wr, *dobs)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if obs_cfg:
            obs_fusion.train()
        if f4_on:
            f4.train()
        logs, parts_log = [], []
        for batch in loader:
            batch = [b.to(device) for b in batch]     # 7 base + K개 dense(obs일 때)
            loss, parts = forward_train(*batch)
            opt.zero_grad(); loss.backward(); opt.step()
            if sched:
                sched.step()
            logs.append(loss.item()); parts_log.append(parts)
        model.eval()
        if obs_cfg:
            obs_fusion.eval()
        if f4_on:
            f4.eval()
        with torch.no_grad():
            val_loss, val_parts = forward(*val_t)
        val_loss = val_loss.item()
        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_f4 = ({k: v.detach().cpu().clone()
                        for k, v in f4.state_dict().items()} if f4_on else None)
        else:
            patience += 1
        if wb:
            wb.log({"epoch": ep, "train/total": np.mean(logs),
                    "val/total": val_loss,
                    **{f"train/{k}": np.mean([x[k] for x in parts_log])
                       for k in parts_log[0]},
                    **{f"val/{k}": v for k, v in val_parts.items()}})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"ep {ep:3d} | train {np.mean(logs):.4f} | val {val_loss:.4f} "
                  f"({val_parts}) | patience {patience}")
        if patience >= t_cfg["early_stop_patience"]:
            print(f"early stop @ ep {ep}")
            break
    print(f"학습 {time.time()-t0:.0f}s, best val {best_val:.4f}")
    model.load_state_dict(best_state)
    if f4_on and best_f4 is not None:
        f4.load_state_dict(best_f4)

    # ---- 평가 ----
    model.eval()
    if obs_cfg:
        obs_fusion.eval()
    if f4_on:
        f4.eval()
    with torch.no_grad():
        zp_v, zc_v = val_t[0], val_t[1]                # 융합 z 원본 (g/h 는 아래에서 val_t[1] 그대로 사용)
        wr_v = val_t[6] if use_wrist else None
        _zd = zp_v.shape[-1]                           # 융합 폭 (S1b=2048); 슬라이스 전에 확정
        if cond_dim:                                   # S1b: 조건 토큰만 SigLIP2 서브블록[0:cond_dim]
            zp_v, zc_v = zp_v[:, :cond_dim], zc_v[:, :cond_dim]
            if use_wrist:
                wr_v = wr_v[:, :cond_dim]
        base = [zp_v, zc_v, val_t[3]] + ([val_t[5]] if use_lang else []) \
            + ([wr_v] if use_wrist else [])
        toks = list(base)
        if obs_cfg:                                   # F3: 학습과 동일하게 관측 토큰 추가
            obs_tok = obs_fusion({n: d for n, d in zip(enc_names, val_t[7:])})
            toks = toks + [obs_tok[:, k] for k in range(K)]
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        # concat 융합/S1b: 좁은 조건 토큰(lang/wrist/SigLIP2 z)을 z 폭(2048)의 SigLIP2 서브블록에
        toks = [t if t.shape[-1] == _zd else           #   zero-pad (기존 런=no-op)
                torch.nn.functional.pad(t, (0, _zd - t.shape[-1])) for t in toks]
        zeta = model(torch.stack(toks, dim=1), generator=gen) if is_flow \
            else model(torch.stack(toks, dim=1))
        lat_target = ae.g(val_t[4], val_t[1])
        ahat_t = ae.h(zeta, val_t[1])
        if f4_on:                                     # ζ_f를 noise flow로 생성(폐루프 동형) + fine 잔차
            zeta_f = f4.flow_sample(torch.cat(base, dim=1), generator=gen)
            ahat_t = ahat_t + f4.fine_action(zeta, zeta_f, val_t[1])
        ahat = ahat_t.cpu().numpy()
    zeta_np = zeta.cpu().numpy()
    lat_np = lat_target.cpu().numpy()
    wm_np = (val_t[2] - val_t[1]).cpu().numpy()
    csim = lambda a, b: float(np.mean((a*b).sum(1) /
        (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1) + 1e-8)))
    lat_cos, wm_cos = csim(zeta_np, lat_np), csim(zeta_np, wm_np)
    Cf = Cf_va
    act_r2 = r2(Cf.reshape(len(Cf), -1), ahat.reshape(len(ahat), -1))
    # 물리 지표(MAE/그리퍼)는 시간영역으로 되돌려 계산 (repr와 무관하게 비교 가능)
    gt = chunkrep.from_repr(Cf, repr_kind) * a_std + a_mean
    pr = chunkrep.from_repr(ahat, repr_kind) * a_std + a_mean
    # 액션 배열: aloha 14D(관절 rad, 그리퍼 [6,13]) / 그 외(예: LIBERO 7D — 마지막이 그리퍼)
    if act_dim == 14:
        arm = [0,1,2,3,4,5,7,8,9,10,11,12]
        grip, g_thr, unit = [6, 13], 0.5, 180/np.pi   # deg 환산
    else:
        arm = list(range(act_dim - 1))
        grip, g_thr, unit = [act_dim - 1], 0.0, 1.0   # 원단위
    mae_deg = float(np.abs(pr[:,:,arm]-gt[:,:,arm]).mean()*unit)
    grip_acc = float(((pr[:,:,grip]>g_thr)==(gt[:,:,grip]>g_thr)).mean()*100)
    # 평균붕괴 진단: 샘플별 오차의 변동계수 (높으면 특정 문맥에서 붕괴 의심)
    per_err = np.abs(pr[:,:,arm]-gt[:,:,arm]).mean(axis=(1,2))
    collapse_cv = float(per_err.std() / (per_err.mean() + 1e-9))
    print(f"\n=== 정책 평가 ({len(Cf)} samples) ===")
    print(f"관절 MAE {mae_deg:.2f}°/step | 그리퍼 {grip_acc:.1f}% | 액션 R² {act_r2:+.3f}")
    print(f"잠재 cos: vs g타깃 {lat_cos:+.3f} / vs Δz타깃 {wm_cos:+.3f} | 붕괴CV {collapse_cv:.2f}")

    metrics = {"score": -mae_deg,   # 그리드 랭킹용 (높을수록 좋게 부호 반전)
               "mae_deg": mae_deg, "grip_acc": grip_acc, "action_r2": act_r2,
               "lat_cos": lat_cos, "wm_cos": wm_cos, "collapse_cv": collapse_cv,
               "best_val_loss": float(best_val), "n_params": n_params,
               "n_val": len(Cf)}
    ckpt_path = Path(os.path.expanduser(t_cfg["checkpoint"]))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg, "metrics": metrics,
                "obs_fusion": obs_fusion.state_dict() if obs_cfg else None,
                "f4": best_f4 if f4_on else None},
               ckpt_path)
    print(f"저장: {ckpt_path}")
    if args.tag:
        out = WS / "outputs" / "grid"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.tag}.json").write_text(json.dumps(
            {"tag": args.tag, "overrides": args.set, **metrics}, indent=1))
    if wb:
        wb.summary.update(metrics)
        wb.finish()


if __name__ == "__main__":
    main()
