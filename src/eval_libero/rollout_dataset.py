"""LIBERO GT 데모 전체 시계열 추론 평가 — 7D 액션 그래프.

  t = 16, 24, ... 마다 (z_{t−16}, z_t, g(A_past)) → f → h → Â_{t:t+16}
  앞 8스텝씩 이어붙여 전체 예측 궤적 구성 (GT 이미지 사용 = 개루프)
  7D = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper] — 3그룹으로 플롯

사용 (clip_libero env):
  python src/eval_libero/rollout_dataset.py                 # val 첫 데모
  python src/eval_libero/rollout_dataset.py --episode 3     # val 3번째 데모
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os

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
matplotlib.rcParams.update({"font.family": ["Noto Sans CJK JP", "sans-serif"],
                            "axes.unicode_minus": False})
import matplotlib.pyplot as plt

from core import chunkrep
from core.anchor import get_anchor
from data.libero import LiberoDataset
from models.networks import DeltaAE, DualDeltaAE
from models.policy import build_policy_from_cfg

WS = Path(__file__).resolve().parents[2]
DIM_NAMES = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]


def load_models(cfg, device):
    ck1 = torch.load(os.path.expanduser(cfg["phase1_ckpt"]),
                     map_location="cpu", weights_only=False)
    p1 = ck1["config"]
    if ck1.get("dual_stream"):
        return _load_models_dual(cfg, device, ck1)
    ae = DeltaAE(ck1["action_dim"], ck1["n_chunk"], p1["model"]["latent_dim"],
                 p1["model"]["hidden"], p1["model"]["layers"],
                 p1["model"]["dropout"],
                 p1["model"].get("state_cond", True),
                 p1["model"].get("decoder_state_cond"),
                 p1["model"].get("encoder_state_cond"),
                 # hybrid phase1(HY03) logit_scale 파라미터 정합 — dz면 no-op(기존 동형).
                 align_mode=p1["model"].get("align_mode", "dz"),
                 contrast_w=float(p1.get("loss", {}).get("contrast", 0.0)),
                 contrast_loss=p1["model"].get("contrast_loss", "infonce"),
                 contrast_head=p1["model"].get("contrast_head", False),
                 sigmoid_bias0=p1["model"].get("sigmoid_bias0", -5.5),
                 align_block=p1["model"].get("align_block"),
                 h_mode=p1["model"].get("h_mode", "mlp"),
                 h_flow_steps=p1["model"].get("h_flow_steps", 5),
                 # capacity sweep ckpt 호환 (구 ckpt 는 키 부재 → None = 비트 동형)
                 hidden_g=p1["model"].get("hidden_g"),
                 hidden_h=p1["model"].get("hidden_h")).to(device).eval()
    ae.load_state_dict(ck1["state_dict"])
    ck2 = torch.load(os.path.expanduser(cfg["train"]["checkpoint"]),
                     map_location="cpu", weights_only=False)
    m = ck2["config"]["module"]
    use_lang = m.get("lang_token", False)
    use_wrist = m.get("wrist_token", False)

    # ---- F3 관측 융합 (module.obs 있을 때만; 없으면 no-obs 기존 경로와 완전 동일) ----
    # phase2 학습(train_phase2.py)의 앵커·ObsFusion 구성을 그대로 미러링한다.
    obs_anchors, obs_fusion, K = None, None, 0
    if m.get("obs"):
        from core.anchor import get_anchor
        from models.obs_fusion import ObsFusion
        obs_cfg = m["obs"]
        obs_anchors, enc_dims = [], {}
        for enc in obs_cfg["encoders"]:
            anc = get_anchor({"anchor": enc})
            if enc["name"] == "siglip2":
                anc.save_tokens = True                # siglip2: 패치 토큰 반환 활성화
            obs_anchors.append((enc["name"], anc,
                                enc.get("camera", "agentview_rgb")))
            enc_dims[enc["name"]] = anc.patch_dim     # dinov2=1024 / siglip2=1152
        obs_fusion = ObsFusion(enc_dims, d_attn=obs_cfg.get("d_attn", 768),
                               n_query=obs_cfg.get("n_query", 8),
                               out_dim=p1["model"]["latent_dim"],
                               pool=obs_cfg.get("pool", "attn"),
                               unshuffle=obs_cfg.get("unshuffle", 1))
        obs_fusion.load_state_dict(ck2["obs_fusion"])
        obs_fusion = obs_fusion.to(device).eval()
        K = obs_fusion.n_query                        # attn: n_query / mean: 1

    # ---- F5-H L1 그리드 관측 (module.grid_obs 있을 때만; 없으면 no-grid 기존 경로와 완전 동일) ----
    #  train_phase2.py 의 grid_obs 구성을 그대로 미러링(UNGATED, 게이트 없음).
    grid_anchor, grid_obs, Kg = None, None, 0
    if m.get("grid_obs") and ck2.get("grid_obs") is not None:
        from core.anchor import get_anchor
        from models.obs_fusion import GridObs
        grid_cfg = m["grid_obs"]
        genc = grid_cfg["anchor"]
        ganc = get_anchor({"anchor": genc})
        if genc["name"] == "siglip2":
            ganc.save_tokens = True
        grid_anchor = (genc["name"], ganc,
                       grid_cfg.get("camera", genc.get("camera", "agentview_rgb")))
        grid_obs = GridObs(patch_dim=ganc.patch_dim,
                           out_dim=p1["model"]["latent_dim"],
                           n_tokens=grid_cfg.get("n_tokens", 16),
                           pool=grid_cfg.get("pool", "avg"),
                           d_attn=grid_cfg.get("d_attn", 768),
                           heads=grid_cfg.get("heads", 8),
                           # W-A(N1) guarded 옵션 미러 (VERIFY A1) — ln 은 state_dict 키에
                           # 필요, tok/group_drop 은 eval()에서 비활성, init_std 는 로드로 덮임.
                           ln=grid_cfg.get("ln", False),
                           tok_drop=grid_cfg.get("tok_drop", 0.0),
                           group_drop=grid_cfg.get("group_drop", 0.0),
                           init_std=grid_cfg.get("init_std"))
        grid_obs.load_state_dict(ck2["grid_obs"])
        grid_obs = grid_obs.to(device).eval()
        Kg = grid_obs.n_tokens

    # ---- P-B LangSelPool (module.patch_obs 있을 때만; 없으면 no-patch 기존 경로와 완전 동일) ----
    #  train_phase2.py 의 patch_obs 구성을 미러링(UNGATED, lang-쿼리 cross-attn). 차원은
    #  저장 가중치에서 직접 도출(f4 미러) — 앵커 patch_dim 오염(1152 함정)과 독립적으로 정확.
    patch_anchor, patch_obs, Kp = None, None, 0
    if m.get("patch_obs") and ck2.get("patch_obs") is not None:
        from core.anchor import get_anchor
        from models.obs_fusion import LangSelPool
        p_cfg = m["patch_obs"]
        penc = p_cfg["anchor"]
        panc = get_anchor({"anchor": penc})
        if penc["name"] == "siglip2":
            panc.save_tokens = True
        patch_anchor = (penc["name"], panc,
                        p_cfg.get("camera", penc.get("camera", "agentview_rgb")))
        psd = ck2["patch_obs"]
        patch_obs = LangSelPool(patch_dim=psd["kv_proj.weight"].shape[1],
                                text_dim=psd["text_q.weight"].shape[1],
                                out_dim=p1["model"]["latent_dim"],
                                n_patch=psd["pos_emb"].shape[0],
                                n_tokens=psd["query_offset"].shape[0],
                                d_attn=p_cfg.get("d_attn", 768),
                                heads=p_cfg.get("heads", 8),
                                tok_drop=p_cfg.get("tok_drop", 0.1),
                                group_drop=p_cfg.get("group_drop", 0.1))
        patch_obs.load_state_dict(psd, strict=True)
        patch_obs = patch_obs.to(device).eval()
        Kp = patch_obs.n_tokens
        # 롤아웃 patch_dim 함정 이중 방어 (DESIGN_WD_WAprime §3.2): 앵커 실폭 == ckpt kv 입력폭
        assert panc.patch_dim == psd["kv_proj.weight"].shape[1], \
            (f"patch_obs: 앵커 patch_dim {panc.patch_dim} != ckpt kv_proj 입력폭 "
             f"{psd['kv_proj.weight'].shape[1]} — 앵커/캐시 키 불일치")

    # ---- W-B Δ̄w-token (grid_obs.wrist_delta=true 일 때만; 없으면 wdelta=None = 기존 동형) ----
    #  N2 재척도 스칼라(train 통계)는 phase2 ckpt dict "wrist_delta_std" 에서 복원 —
    #  rollout 은 grid 인코딩(사영 전 pool2 patch-mean)에서 w_tok 을 유도(추가 인코딩 0회).
    wdelta = None
    if grid_obs is not None and m["grid_obs"].get("wrist_delta", False):
        wds = ck2.get("wrist_delta_std")
        assert wds is not None, \
            "wrist_delta: phase2 ckpt 에 wrist_delta_std(N2 스칼라) 없음 — 재학습 필요"
        wdelta = {"scale": float(wds["scale"]),
                  "sigma_ref": float(wds["sigma_ref"]),
                  "sigma_delta": float(wds["sigma_delta"])}

    policy = build_policy_from_cfg(
        m, n_tokens=3 + int(use_lang) + int(use_wrist) + K + Kg + Kp
        + int(wdelta is not None),
        latent_dim=p1["model"]["latent_dim"],
        action_flat_dim=ck1["n_chunk"] * ck1["action_dim"]).to(device).eval()
    policy.load_state_dict(ck2["state_dict"])
    wrist_cam = ck2["config"]["data"].get("wrist_camera") if use_wrist else None

    # ---- C1/F4 fine 채널 (module.f4.enable + 체크포인트에 f4 state 존재 시에만) ----
    # f4 is None 이면 롤아웃의 fine 경로가 완전 게이트됨 → pooled-only와 비트 동형.
    # dense(patch ΔF)/text 차원은 저장 가중치에서 직접 도출 → 추론 시 dense 인코더 로드
    # 불필요(ζ_f는 base 조건 noise-flow로 생성되며 미래 patch ΔF를 쓰지 않음).
    f4 = None
    f4_cfg = m.get("f4")
    if f4_cfg and f4_cfg.get("enable") and ck2.get("f4") is not None:
        from models.f4 import build_f4_from_cfg
        fsd = ck2["f4"]
        n_base = 3 + int(use_lang) + int(use_wrist)    # f4 flow 조건 = base 토큰
        fine_mode = f4_cfg.get("fine_mode", "kquery")
        if fine_mode == "paramfree":
            # A1 팔: readout.weight (bneck, dense_dim+n_patch). encode는 추론에 미사용이나
            # strict 로드 위해 shape 재현 필요. paramfree는 dense_dim/n_patch를 개별로 쓰지
            # 않고 합(=readout 입력)만 쓰므로 dense_dim=합·n_patch=0 로 동형 재구성.
            f4 = build_f4_from_cfg(
                f4_cfg,
                dense_dim=fsd["readout.weight"].shape[1], text_dim=1,
                latent_dim=p1["model"]["latent_dim"],
                n_base_tokens=n_base, action_dim=ck1["action_dim"],
                n_chunk=ck1["n_chunk"], n_patch=0)
        else:
            f4 = build_f4_from_cfg(
                f4_cfg,
                dense_dim=fsd["kv_proj.weight"].shape[1],   # dense patch 차원(학습 인코더)
                text_dim=fsd["text_q.weight"].shape[1],     # 쿼리 초기화 텍스트 차원
                latent_dim=p1["model"]["latent_dim"],
                n_base_tokens=n_base, action_dim=ck1["action_dim"],
                n_chunk=ck1["n_chunk"], n_patch=fsd["pos_emb"].shape[0])
        f4.load_state_dict(fsd, strict=True)
        f4 = f4.to(device).eval()

    # obs_anchors/obs_fusion/f4/grid_obs/patch_obs 는 해당 모듈일 때만 non-None; 미사용 호출부는 *_로 무시.
    # dual=None (단일 스트림). grid_anchor/grid_obs/wdelta 는 14·15·16번째,
    # patch_anchor/patch_obs 는 17·18번째 (뒤에 append → *_ 호출부 안전).
    return (ae, policy, ck1["a_mean"], ck1["a_std"], ck1["n_chunk"],
            ck1["action_dim"], use_lang, ck1.get("chunk_repr", "time"),
            wrist_cam, obs_anchors, obs_fusion, f4, None, grid_anchor, grid_obs,
            wdelta, patch_anchor, patch_obs)


def _load_models_dual(cfg, device, ck1):
    """dual-stream 모델 로드 (DualDeltaAE + FlowPolicy). load_models 와 동일 arity의
    18-튜플 반환 — 13번째 dual dict 로 caller(rollout)가 wrist 앵커/dim_cat/디코딩 분기.
    obs/f4/aug 는 dual 미지원 → 모두 None. wrist_cam 은 손목 프레임 인코딩용으로 반환.
    W-C: stream_standardize(N3)는 phase1 config 로 재구성(buffer 는 state_dict 복원),
    x0_per_dim(N4)은 build_policy_from_cfg 가 module config 에서 자동 반영,
    wrist_cond_sig(결함⑤ 격리)는 dual dict 로 caller 에 전달(토큰열 분기)."""
    p1 = ck1["config"]
    dim_main, dim_wrist = ck1["dim_main"], ck1["dim_wrist"]
    dc = dim_main + dim_wrist
    ae = DualDeltaAE(ck1["action_dim"], ck1["n_chunk"], dim_main, dim_wrist,
                     p1["model"]["hidden"], p1["model"]["layers"],
                     p1["model"]["dropout"],
                     p1["model"].get("state_cond", True),
                     stream_standardize=p1["model"].get(
                         "stream_standardize", False)).to(device).eval()
    ae.load_state_dict(ck1["state_dict"])
    ck2 = torch.load(os.path.expanduser(cfg["train"]["checkpoint"]),
                     map_location="cpu", weights_only=False)
    m = ck2["config"]["module"]
    use_lang = m.get("lang_token", False)
    wc_sig = bool(m.get("wrist_cond_sig", False))   # W-C: [zp,zc,a_emb,zw_sig](+lang)=4+lang
    policy = build_policy_from_cfg(m, n_tokens=(4 if wc_sig else 5) + int(use_lang),
                                   latent_dim=dc).to(device).eval()
    policy.load_state_dict(ck2["state_dict"])
    wrist_cam = ck2["config"]["data"].get("wrist_camera")
    assert wrist_cam, "dual_stream: data.wrist_camera 필요"
    dual = {"dim_main": dim_main, "dim_wrist": dim_wrist, "dim_cat": dc,
            "anchor_wrist": ck2["config"]["anchor_wrist"],
            "wrist_cond_sig": wc_sig}
    # use_lang 만 의미; use_wrist/obs/f4/grid/wdelta/patch 는 dual 경로에서 미사용(None).
    return (ae, policy, ck1["a_mean"], ck1["a_std"], ck1["n_chunk"],
            ck1["action_dim"], use_lang, ck1.get("chunk_repr", "time"),
            wrist_cam, None, None, None, dual, None, None, None, None, None)


def sample_zeta(policy, f4, tokens, generator=None):
    """ζ_g (정책) + (f4 있으면) ζ_f 를 하나의 적분 루프·공유 τ로 함께 샘플.

    cowork §D3-A 조건②: ζ_g flow와 ζ_f flow는 별개의 독립 적분이 아니라 단일
    통합 샘플링 루프에서 동일 τ 스케줄로 전진해야 한다. 두 속도장은 서로의 현재
    상태에 의존하지 않으므로(각자 ctx·x만) 통합 루프의 결과는 각자 독립 적분과
    수치적으로 동일하며, 학습부(train_phase2)의 순차 2-루프와도 전송의미 동등.

    f4 is None 이면 policy 단독 샘플(policy.forward)과 완전히 동일한 호출 → 비트 동형.
    generator: rollout --flow-noise-mode locked 전용 주입(정책 x0·ζ_f 노이즈 잠금).
    None(기본)이면 종전과 동일하게 전역 RNG 사용 — 기존 호출부 비트 동형.
    """
    if f4 is None:
        if generator is None:
            return policy(tokens), None                 # 현행 호출과 동일 = 비트 동형
        return policy(tokens, generator=generator), None  # locked: FlowPolicy에만 전달
    assert policy.steps == f4.flow_steps, (
        f"shared-τ 요구: policy.steps({policy.steps}) == "
        f"f4.flow_steps({f4.flow_steps})")
    base_flat = tokens.flatten(1)                       # C1: tokens == base(관측 없음)
    ctx_g = policy.ctx(base_flat)
    x_g = policy._x0(tokens, generator)                 # source=past → a_emb(RNG 무소비)
    ctx_f = f4.ctx(base_flat)                           # ζ_f 조건 = 동일 base 토큰
    x_f = torch.randn((len(tokens), f4.zf_dim), device=tokens.device,
                      generator=generator) * f4.zf_std  # ζ_f source=noise(미래 무접근)
    dt = 1.0 / policy.steps
    for i in range(policy.steps):                       # 단일 루프 · 공유 τ = i·dt
        t = torch.full((len(x_g), 1), i * dt, device=x_g.device)
        x_g = x_g + policy._v(x_g, ctx_g, t) * dt
        x_f = x_f + f4._v(x_f, ctx_f, t) * dt
    return x_g, x_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(WS / "configs" / "phase2_libero.yaml"))
    ap.add_argument("--episode", type=int, default=0, help="val 분할 내 인덱스")
    ap.add_argument("--exec-horizon", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (ae, policy, a_mean, a_std, n_chunk, act_dim, use_lang,
     repr_kind, wrist_cam, obs_anchors, obs_fusion, _f4, dual,
     grid_anchor, grid_obs, wdelta, patch_anchor, patch_obs) = load_models(cfg, device)
    ds = LiberoDataset(cfg)
    clip = get_anchor(cfg)          # 앵커 config 반영 (무-anchor면 ClipAnchor=ClipWrapper와 동일)
    clip_wrist = get_anchor({"anchor": dual["anchor_wrist"]}) if dual else None

    eps = ds.episode_files()
    rng = np.random.RandomState(cfg["train"]["seed"])
    perm = rng.permutation(len(eps))
    v = cfg["data"]["val_episodes"]
    n_val = max(1, round(len(eps) * v)) if v < 1 else int(v)
    ep = eps[perm[args.episode % n_val]]
    print(f"에피소드: {ds._key(ep)}")
    print(f"지시문: {ds.instruction(ep)}")

    acts = ds.load_actions(ep)
    Z = ds.embeddings(clip, ep)
    # dual: 손목 변위 스트림은 별도 anchor_wrist pooled (prev/cur); 단일: 손목 토큰(main clip).
    Zw = ds.embeddings(clip_wrist if dual else clip, ep, wrist_cam) \
        if (wrist_cam or dual) else None
    # W-C: 조건용 SigLIP2-wrist cur (main 앵커; DINOv3 Zw 는 h 상태 전용으로 유지)
    Zw_sig = (ds.embeddings(clip, ep, wrist_cam)
              if dual and dual.get("wrist_cond_sig") else None)
    # F3: 앵커별 dense patch 토큰 D[t] (Z[t]=z_cur 와 동일 인덱스 정렬)
    D_obs = ([(name, ds.dense_embeddings(anc, ep, cam))
              for name, anc, cam in obs_anchors] if obs_fusion is not None else [])
    # F5-H L1: 그리드 관측 앵커(DINOv3) dense patch 격자 D[t] (Z[t] 와 동일 인덱스 정렬)
    D_grid = (ds.dense_embeddings(grid_anchor[1], ep, grid_anchor[2])
              if grid_obs is not None else None)
    # P-B: LangSelPool 앵커 dense patch 격자 D[t] (Z[t] 와 동일 인덱스 정렬)
    D_patch = (ds.dense_embeddings(patch_anchor[1], ep, patch_anchor[2])
               if patch_obs is not None else None)
    lang = torch.tensor(ds.instruction_embedding(clip, ep)[None],
                        device=device) if use_lang else None
    T = len(acts)
    span, H = ds.span, args.exec_horizon

    def norm(a):
        return ((a - a_mean) / a_std).astype(np.float32)

    pred = np.full_like(acts, np.nan)
    t = span
    with torch.no_grad():
        while t + span <= T:
            z_prev = torch.tensor(Z[t - span][None], device=device)
            z_cur = torch.tensor(Z[t][None], device=device)
            past = chunkrep.to_repr(
                norm(ds.resample_chunk(acts[t - span:t])), repr_kind)[None]
            past_t = torch.tensor(past, device=device)
            if dual:                                  # dual-stream 변위 정책
                dcw = dual["dim_cat"]
                zwp = torch.tensor(Zw[t - span][None], device=device)
                zwc = torch.tensor(Zw[t][None], device=device)
                a_emb = ae.encode(past_t, z_prev, zwp)         # concat ζ_past (dc)
                _pad = lambda x: torch.nn.functional.pad(x, (0, dcw - x.shape[-1]))
                if dual.get("wrist_cond_sig"):        # W-C: 조건=[zp,zc,a_emb,zw_sig](+lang)
                    zw_sig = torch.tensor(Zw_sig[t][None], device=device)
                    toks = [_pad(z_prev), _pad(z_cur), a_emb, _pad(zw_sig)] \
                        + ([_pad(lang)] if use_lang else [])
                else:                                 # 기존 dual 6토큰 (비트 동형)
                    toks = [_pad(z_prev), _pad(z_cur), a_emb, _pad(zwp), _pad(zwc)] \
                        + ([_pad(lang)] if use_lang else [])
                zeta = policy(torch.stack(toks, dim=1))
                ahat = chunkrep.from_repr(
                    ae.decode(zeta, z_cur, zwc).cpu().numpy()[0],
                    repr_kind) * a_std + a_mean
            else:
                a_emb = ae.g(past_t, z_prev)
                toks = [z_prev, z_cur, a_emb] + ([lang] if use_lang else []) \
                    + ([torch.tensor(Zw[t][None], device=device)]
                       if wrist_cam else [])
                if obs_fusion is not None:            # F3: 관측 토큰 K개를 열 끝에 추가
                    obs_tok = obs_fusion({name: torch.tensor(D[t][None], device=device)
                                          for name, D in D_obs})   # (1,K,768)
                    toks = toks + [obs_tok[:, k] for k in range(obs_tok.size(1))]
                if grid_obs is not None:              # F5-H L1: UNGATED 그리드 토큰 Kg개를 열 끝에 추가
                    grid_tok = grid_obs(torch.tensor(D_grid[t][None], device=device))
                    toks = toks + [grid_tok[:, k] for k in range(grid_tok.size(1))]
                if patch_obs is not None:             # P-B: 언어-선택 patch 토큰 Kp개 — canonical 마지막
                    patch_tok = patch_obs(
                        torch.tensor(D_patch[t][None], device=device), lang)
                    toks = toks + [patch_tok[:, k] for k in range(patch_tok.size(1))]
                if wdelta is not None:                # W-B w_tok #10 — canonical 마지막 (A2)
                    wd = ((D_grid[t].mean(0) - D_grid[max(t - span, 0)].mean(0))
                          * wdelta["scale"]).astype(np.float32)
                    toks = toks + [torch.tensor(wd[None], device=device)]
                zeta = policy(torch.stack(toks, dim=1))
                ahat = chunkrep.from_repr(ae.h(zeta, z_cur).cpu().numpy()[0],
                                          repr_kind) * a_std + a_mean
            n_exec = min(H, T - t)
            pred[t:t + n_exec] = ahat[:n_exec]
            t += H

    valid = ~np.isnan(pred[:, 0])
    mae = np.abs(pred[valid] - acts[valid]).mean(axis=0)
    grip_acc = ((pred[valid][:, 6] > 0) == (acts[valid][:, 6] > 0)).mean() * 100
    print(f"MAE(정규화 [-1,1] 단위): pos {mae[:3].mean():.3f} | "
          f"rot {mae[3:6].mean():.3f} | 그리퍼 정확도 {grip_acc:.1f}% "
          f"| 추론 구간 {valid.sum()}/{T}")

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), dpi=110, sharex=True)
    tt = np.arange(T) / 20.0
    for d in range(act_dim):
        ax = axes[d]
        ax.plot(tt, acts[:, d], color="#4477AA", lw=1.6, label="GT")
        ax.plot(tt, pred[:, d], color="#EE6677", lw=1.2, ls="--", label="예측")
        ax.set_title(DIM_NAMES[d], fontsize=9)
        ax.grid(color="#E5E7EB", lw=0.5)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f'{ds._key(ep)}\n"{ds.instruction(ep)}"\n'
                 f'pos MAE {mae[:3].mean():.3f} | rot {mae[3:6].mean():.3f} | '
                 f'그리퍼 {grip_acc:.1f}%', fontsize=11)
    fig.tight_layout()
    out = WS / "outputs" / "eval" / f"rollout_dataset_libero_{ds._key(ep)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
