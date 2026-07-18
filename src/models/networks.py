"""Phase 1 — 액션청크 <-> Δz(pooled 768) 결합 AE.

  인코더 g: 액션청크(T×D) + z_t → ζ (768)     [1D-CNN, 상태조건]
  디코더 h: Δz(768) + z_t → 액션청크           [MLP, 상태조건]

손실 (VITA 동형):
  align: g(a, z_t) ≈ Δz  (MSE + 0.5·cos)      — FM 자리
  recon: h(Δz, z_t) ≈ a  (L1)                  — FLD 대응
  cycle: h(g(a,z_t), z_t) ≈ a  (L1)            — L_AE 대응, phase2 디코딩 경로
"""
import torch
import torch.nn as nn


class ChunkEncoder(nn.Module):
    """(B, T, D) [+ z_t] → (B, latent)"""

    def __init__(self, action_dim, latent_dim=768, hidden=512, layers=4,
                 dropout=0.0, state_cond=True):
        super().__init__()
        self.state_cond = state_cond
        convs, c_in = [], action_dim
        for _ in range(layers):
            convs += [nn.Conv1d(c_in, hidden, kernel_size=3, padding=1),
                      nn.GELU()]
            if dropout > 0:
                convs.append(nn.Dropout(dropout))
            c_in = hidden
        self.conv = nn.Sequential(*convs)
        head_in = hidden + (latent_dim if state_cond else 0)
        self.head = nn.Sequential(nn.Linear(head_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, latent_dim))

    def forward(self, chunk, z_t=None):
        x = self.conv(chunk.transpose(1, 2)).mean(dim=2)   # 시간축 평균 풀링
        if self.state_cond:
            x = torch.cat([x, z_t], dim=1)
        return self.head(x)


class ChunkDecoder(nn.Module):
    """(B, latent) [+ z_t] → (B, T, D)"""

    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=True):
        super().__init__()
        self.state_cond = state_cond
        in_dim = latent_dim * (2 if state_cond else 1)
        dims = [in_dim] + [hidden] * (layers - 1)
        mlp = [nn.LayerNorm(in_dim)]
        for i in range(len(dims) - 1):
            mlp += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
        mlp.append(nn.Linear(dims[-1], n_chunk * action_dim))
        self.mlp = nn.Sequential(*mlp)
        self.n_chunk, self.action_dim = n_chunk, action_dim

    def forward(self, z, z_t=None):
        if self.state_cond:
            z = torch.cat([z, z_t], dim=1)
        return self.mlp(z).view(-1, self.n_chunk, self.action_dim)


class ChunkFlowDecoder(nn.Module):
    """(B, latent) [+ z_t] → (B, T, D)  — 조건부 flow-matching 디코더 (VITA FLD 완성).

    결정론 MLP(ChunkDecoder)가 뭉개는 p(action | ζ, z_t)의 미세 다봉성을 분포로 모델링:
    velocity field v(a_τ, τ | cond=[ζ, z_t]) 를 CFM으로 학습, 노이즈 x0 → 액션청크를 Euler K스텝 적분.
    x0_std 버퍼 = 학습 시작 시 액션청크 타깃 std로 설정(FlowPolicy.x0_std 동형 — 스케일 정합).
    forward(z, z_t) 는 ChunkDecoder 와 동일 인터페이스(샘플 반환)라 phase2/rollout 무변경 호환.
    """

    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=True, steps=5):
        super().__init__()
        self.state_cond = state_cond
        self.n_chunk, self.action_dim, self.steps = n_chunk, action_dim, steps
        self.a_dim = n_chunk * action_dim
        cond_in = latent_dim * (2 if state_cond else 1)
        self.cond = nn.Sequential(nn.LayerNorm(cond_in),
                                  nn.Linear(cond_in, hidden), nn.GELU())
        self.t_embed = nn.Sequential(nn.Linear(1, 128), nn.GELU(),
                                     nn.Linear(128, 128))
        net = [nn.Linear(self.a_dim + hidden + 128, hidden), nn.GELU()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.GELU()]
            if dropout > 0:
                net.append(nn.Dropout(dropout))
        net.append(nn.Linear(hidden, self.a_dim))
        self.v = nn.Sequential(*net)
        self.register_buffer("x0_std", torch.ones(1))

    def _ctx(self, z, z_t):
        c = torch.cat([z, z_t], dim=1) if self.state_cond else z
        return self.cond(c)

    def _vel(self, a, ctx, t):
        return self.v(torch.cat([a, ctx, self.t_embed(t)], dim=1))

    def forward(self, z, z_t=None, generator=None):      # 샘플 (ChunkDecoder 호환)
        ctx = self._ctx(z, z_t)
        x = torch.randn((len(z), self.a_dim), device=z.device,
                        generator=generator) * self.x0_std
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((len(x), 1), i * dt, device=x.device)
            x = x + self._vel(x, ctx, t) * dt
        return x.view(-1, self.n_chunk, self.action_dim)

    def cfm_loss(self, chunk, z, z_t=None, generator=None):
        """학습용 CFM 손실 — recon/cycle 의 L1 대체. target = 액션청크(평탄화)."""
        ctx = self._ctx(z, z_t)
        target = chunk.reshape(len(chunk), -1)
        x0 = torch.randn_like(target) * self.x0_std
        t = torch.rand((len(target), 1), device=target.device, generator=generator)
        xt = (1 - t) * x0 + t * target
        return nn.functional.mse_loss(self._vel(xt, ctx, t), target - x0)


class ResidualFlowDecoder(nn.Module):
    """h_mode="residual_flow" — MLP 조건평균(결정론 백본) + 잔차 flow (콜리그 M7/Q2 동형).

    ┌ 왜 전면교체 h-flow(37%)와 다른가 ────────────────────────────────────────┐
    │ 기존 h_mode="flow"는 결정론 MLP를 flow로 **완전 교체** → 재계획(receding-  │
    │ horizon)마다 조건분포에서 독립 모드를 샘플 → 매 계획이 다른 모드로 튀는     │
    │ mode-switching 배회(closed-loop 37%). 여기선 MLP 조건평균이 **결정론 앵커** │
    │ 로 남아 궤적을 고정하고, flow는 평균 주위의 **소분산 잔차**만 모델링한다.   │
    │ 잔차는 작으므로 재계획 간 평균이 지배 → 일관·안정 + 다봉성은 잔차로 유지.   │
    └─────────────────────────────────────────────────────────────────────────┘
      a = mean(ζ,z_t) + res(ζ,z_t)
        mean = ChunkDecoder      (결정론, recon/cycle L1로 학습 — 기존 mlp와 동일)
        res  = ChunkFlowDecoder  (잔차 (a − mean.detach())만 CFM으로 수송)
      res.x0_std = **잔차** std(액션 std 아님, train_phase1이 주입) → 평균 주위 소분산.
    forward(z,z_t[,generator])는 ChunkDecoder/ChunkFlowDecoder와 동일 인터페이스
    (샘플 반환)라 phase2/rollout의 ae.h(ζ,z_cur)[,generator] 무변경 호환.
    generator는 잔차 flow x0 노이즈 재현(rollout --flow-noise-mode walk/locked 안정화)용."""

    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=True, steps=5):
        super().__init__()
        self.n_chunk, self.action_dim = n_chunk, action_dim
        self.mean = ChunkDecoder(action_dim, n_chunk, latent_dim, hidden,
                                 layers, dropout, state_cond)
        self.res = ChunkFlowDecoder(action_dim, n_chunk, latent_dim, hidden,
                                    layers, dropout, state_cond, steps=steps)

    def forward(self, z, z_t=None, generator=None):      # 샘플 = 평균 + 잔차 (ChunkDecoder 호환)
        return self.mean(z, z_t) + self.res(z, z_t, generator=generator)


class DeltaAE(nn.Module):
    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=True,
                 decoder_state_cond=None, encoder_state_cond=None,
                 align_mode="dz", contrast_w=0.0, contrast_loss="infonce",
                 contrast_head=False, sigmoid_bias0=-5.5, align_block=None,
                 h_mode="mlp", h_flow_steps=5):
        """decoder_state_cond/encoder_state_cond: h/g 각각 독립적으로 상태조건을
        끄기 위한 오버라이드. None이면 state_cond와 동일(기존 동작 유지).
          decoder_state_cond=False (C0) : h가 z_t 없이도 되는지 (실측: 거의 무손실)
          encoder_state_cond=False (C1) : g가 z_t 없이 액션만으로 Δz를 예측 가능한지
                                          — "행동만으로 이미지변화가 결정되는가" 검증

        align_mode (C8/HY03 언어정렬): dz(기준·기본) / direct(InfoNCE→모션문장) /
        hybrid(dz + λc·InfoNCE). direct/hybrid도 recon·cycle 유지. 기본 dz에선 아래
        contrastive 파라미터를 만들지 않아 state_dict 불변(기존 CLIP-768 경로 비트 동형)."""
        super().__init__()
        enc_cond = state_cond if encoder_state_cond is None else encoder_state_cond
        dec_cond = state_cond if decoder_state_cond is None else decoder_state_cond
        self.g = ChunkEncoder(action_dim, latent_dim, hidden, layers,
                              dropout, enc_cond)
        # h_mode: mlp(기본, 결정론 조건평균) | flow(S2, 조건부 flow-matching 디코더 — 미세 다봉성)
        #   | residual_flow(콜리그 M7/Q2 동형: MLP 조건평균 + 잔차 flow — 평균이 결정론 앵커,
        #     잔차 flow가 소분산 다봉성을 얹음 → 전면교체 flow의 mode-switching 배회 회피).
        # mlp/flow 분기는 아래 생성 인자·순서가 종전과 동일 → state_dict 비트 동형(regression-0).
        assert h_mode in ("mlp", "flow", "residual_flow"), h_mode
        self.h_mode = h_mode
        if h_mode == "flow":
            self.h = ChunkFlowDecoder(action_dim, n_chunk, latent_dim, hidden,
                                      layers, dropout, dec_cond, steps=h_flow_steps)
        elif h_mode == "residual_flow":
            self.h = ResidualFlowDecoder(action_dim, n_chunk, latent_dim, hidden,
                                         layers, dropout, dec_cond, steps=h_flow_steps)
        else:
            self.h = ChunkDecoder(action_dim, n_chunk, latent_dim, hidden,
                                  layers, dropout, dec_cond)
        assert align_mode in ("dz", "direct", "hybrid"), align_mode
        self.align_mode = align_mode
        # S1b: 융합 ζ(예: dualconcat 2048)의 SigLIP2 블록[0:align_block]만 모션문장 InfoNCE
        #   정렬(텍스트=SigLIP2 dim). None이면 전체 ζ(기존 CLIP-768/SigLIP 단일 경로 동일).
        self.align_block = align_block
        self.contrast_w = contrast_w
        self.contrast_loss = contrast_loss   # "infonce"(기본) | "sigmoid"(SigLIP식)
        if align_mode != "dz":
            import numpy as np
            if contrast_loss == "sigmoid":
                # SigLIP 관례 초기화 (2303.15343): t'=log10, b=sigmoid_bias0 (전역 1쌍)
                self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
                self.logit_bias = nn.Parameter(torch.tensor(float(sigmoid_bias0)))
            else:
                # InfoNCE 학습형 온도 (CLIP 관례: log(1/0.07))
                self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))
            if contrast_head:   # 노름 분리 (SimCLR 투영헤드 원리)
                self.contrast_proj = nn.Linear(latent_dim, latent_dim)

    def info_nce(self, ghat, text_emb, sent_ids):
        """대조 정렬 손실 (direct/hybrid 전용). contrast_loss="infonce"(SupCon 다중양성:
        동일 문장 샘플은 다중 양성으로 마스킹 — 고유 문장 수가 적어 배치 내 중복 흔함) |
        "sigmoid"(SigLIP식 쌍별 이진, 전역 t·b). contrast_proj 존재 시 g를 투영."""
        if self.align_block is not None:         # S1b: SigLIP2 블록만 텍스트 정렬 (기하 블록 제외)
            ghat = ghat[..., :self.align_block]  #   ζ[0:1024] ↔ SigLIP2 모션텍스트(1024) 차원 정합
        if hasattr(self, "contrast_proj"):
            ghat = self.contrast_proj(ghat)
        gn = nn.functional.normalize(ghat, dim=1)
        tn = nn.functional.normalize(text_emb, dim=1)
        pos = (sent_ids[:, None] == sent_ids[None, :])       # (B, B) 동일 문장 = 양성
        if self.contrast_loss == "sigmoid":
            logits = (gn @ tn.T) * self.logit_scale.exp().clamp(max=200.0) \
                + self.logit_bias
            # 양성/음성 각각 평균 후 합산 (스케일 안정: 둘 다 O(1))
            lp = nn.functional.logsigmoid(logits)[pos].mean()
            ln = nn.functional.logsigmoid(-logits)[~pos].mean()
            return -(lp + ln)
        logits = gn @ tn.T * self.logit_scale.exp().clamp(max=100.0)
        all_lse = torch.logsumexp(logits, dim=1)
        pos_lse = torch.logsumexp(
            logits.masked_fill(~pos, float("-inf")), dim=1)
        return (all_lse - pos_lse).mean()

    def losses(self, chunk, delta_z, w, z_t=None, align_type="mse_cos",
               text_emb=None, sent_ids=None):
        """align_type: mse_cos(기본, MSE+0.5·(1−cos)) | l1(순수 L1 — V-JEPA류 월드모델
        문헌에서 가장 흔한 선택, 방향/크기를 분리하지 않고 원시벡터 거리 하나로 처리).
        text_emb/sent_ids: align_mode∈{direct,hybrid}일 때 모션문장 정렬용 (그 외 무시).
        dz 경로(기본)는 아래 total 합산식이 기존과 완전 동일 → 비트 동형 보존."""
        ghat = self.g(chunk, z_t)                # 액션(+상태) → 잠재
        if align_type == "l1":
            l_align = nn.functional.l1_loss(ghat, delta_z)
        else:
            cos = nn.functional.cosine_similarity(ghat, delta_z, dim=1)
            l_align = (nn.functional.mse_loss(ghat, delta_z)
                       + 0.5 * (1 - cos).mean())
        l_res = None                             # residual_flow 전용 잔차 CFM 항 (그 외 None → total 불변)
        if self.h_mode == "flow":                # S2: recon/cycle = 조건부 CFM (L1 대체)
            l_recon = self.h.cfm_loss(chunk, delta_z, z_t)   # p(a | Δz, z_t)
            l_cycle = self.h.cfm_loss(chunk, ghat, z_t)      # p(a | g(a), z_t), grad→g
        elif self.h_mode == "residual_flow":     # M7/Q2 동형: MLP 조건평균(L1) + 잔차 flow(CFM)
            ahat = self.h.mean(delta_z, z_t)     # 결정론 조건평균 = 안정 앵커(전면교체 flow 배회 회피)
            acyc = self.h.mean(ghat, z_t)        # 왕복 (grad→g; 잔차 flow는 g로 grad 안 보냄)
            l_recon = nn.functional.l1_loss(ahat, chunk)     # 평균은 기존 mlp와 동일한 L1
            l_cycle = nn.functional.l1_loss(acyc, chunk)
            # 잔차 target = a − mean.detach(); cond(잠재)도 detach → g/평균 목표 불변, res만 학습.
            l_res = (self.h.res.cfm_loss(chunk - ahat.detach(), delta_z, z_t)
                     + self.h.res.cfm_loss(chunk - acyc.detach(), ghat.detach(), z_t))
        else:
            ahat = self.h(delta_z, z_t)          # 실제 Δz(+상태) → 액션 (FLD 대응)
            acyc = self.h(ghat, z_t)             # 왕복 (phase2 디코딩 경로)
            l_recon = nn.functional.l1_loss(ahat, chunk)
            l_cycle = nn.functional.l1_loss(acyc, chunk)
        if self.align_mode == "direct":         # dz-align 제외 (대조 정렬만)
            total = w["recon"] * l_recon + w["cycle"] * l_cycle
            parts = {"recon": l_recon.item(), "cycle": l_cycle.item()}
        else:                                    # dz / hybrid — 기존 dz 식과 완전 동일
            total = w["align"] * l_align + w["recon"] * l_recon + w["cycle"] * l_cycle
            parts = {"align": l_align.item(), "recon": l_recon.item(),
                     "cycle": l_cycle.item()}
        if self.align_mode in ("direct", "hybrid"):
            assert text_emb is not None and sent_ids is not None, \
                "direct/hybrid 정렬엔 모션 문장 임베딩 필요"
            l_con = self.info_nce(ghat, text_emb, sent_ids)
            cw = self.contrast_w if self.align_mode == "hybrid" else w["align"]
            total = total + cw * l_con
            parts["contrast"] = l_con.item()
        if l_res is not None:                    # residual_flow: 잔차 CFM 항 (w["res_flow"] 기본 1.0)
            total = total + w.get("res_flow", 1.0) * l_res
            parts["res_flow"] = l_res.item()
        return total, parts


class DualDeltaAE(nn.Module):
    """Dual-stream displacement AE (손목캠 = 추론 변위 스트림, cowork dual_stream).

    단일 스트림 DeltaAE(agentview만)와 달리 **카메라별로 분리된** 인코더 2개가
    각자 액션청크(+해당 카메라 상태)를 그 카메라 잠재 변위로 매핑한다:
        g_main (a, z_main_cur) → ζ_main ≈ Δz_main   (agentview, 예: SigLIP2)
        g_wrist(a, z_wrist_cur)→ ζ_wrist≈ Δz_wrist   (wrist,     예: DINOv3 접촉/기하)
    액션은 **두 변위를 함께** 디코딩 — 단일 h가 concat 변위와 concat 상태를 받는다:
        h([ζ_main ; ζ_wrist], [z_main_cur ; z_wrist_cur]) → 액션청크.
    (widened ChunkDecoder 재사용: latent_dim=dim_main+dim_wrist. ζ_main·ζ_wrist가
     각자 Δz_main·Δz_wrist를 근사하므로 concat 잠재폭 == concat 상태폭 = dim_cat →
     ChunkDecoder의 in_dim=dim_cat*2 규약이 그대로 성립. 두 h를 합산하는 대안보다
     상태-변위 교차조건을 한 MLP가 학습하므로 단순·정확.)

    손실 = w_align·(align_main + align_wrist) + w_recon·recon + w_cycle·cycle.
      align_main/wrist: ζ_·≈Δz_· (각 카메라 잠재공간, MSE+0.5·(1−cos) 또는 L1)
      recon/cycle     : 두 변위를 함께 디코딩한 단일 액션청크에 대한 L1 (VITA 동형).
    이 클래스는 dual_stream 경로에서만 인스턴스화된다 — 위 단일 DeltaAE는 무변경
    (비트 동형). h_mode="mlp" 속성은 rollout/phase2의 getattr(ae,"h_mode") 호환용."""

    def __init__(self, action_dim, n_chunk, dim_main, dim_wrist, hidden=512,
                 layers=4, dropout=0.0, state_cond=True):
        super().__init__()
        self.dim_main = int(dim_main)
        self.dim_wrist = int(dim_wrist)
        self.dim_cat = self.dim_main + self.dim_wrist
        self.h_mode = "mlp"                       # rollout/phase2 호환 (분포 디코더 아님)
        self.g_main = ChunkEncoder(action_dim, self.dim_main, hidden, layers,
                                   dropout, state_cond)
        self.g_wrist = ChunkEncoder(action_dim, self.dim_wrist, hidden, layers,
                                    dropout, state_cond)
        self.h = ChunkDecoder(action_dim, n_chunk, self.dim_cat, hidden, layers,
                              dropout, state_cond)

    def encode(self, chunk, z_main, z_wrist):
        """액션청크 → concat 변위 ζ=[ζ_main ; ζ_wrist] (dim_cat). phase2 flow 타깃."""
        return torch.cat([self.g_main(chunk, z_main),
                          self.g_wrist(chunk, z_wrist)], dim=1)

    def decode(self, zeta_cat, z_main, z_wrist):
        """concat 변위(dim_cat) + concat 상태 → 액션청크. phase2/rollout 디코딩 경로."""
        return self.h(zeta_cat, torch.cat([z_main, z_wrist], dim=1))

    @staticmethod
    def _align(ghat, dz, align_type):
        if align_type == "l1":
            return nn.functional.l1_loss(ghat, dz)
        cos = nn.functional.cosine_similarity(ghat, dz, dim=1)
        return nn.functional.mse_loss(ghat, dz) + 0.5 * (1 - cos).mean()

    def losses(self, chunk, dz_main, dz_wrist, z_main, z_wrist, w,
               align_type="mse_cos"):
        gm = self.g_main(chunk, z_main)
        gw = self.g_wrist(chunk, z_wrist)
        l_am = self._align(gm, dz_main, align_type)
        l_aw = self._align(gw, dz_wrist, align_type)
        zt_cat = torch.cat([z_main, z_wrist], dim=1)
        ahat = self.h(torch.cat([dz_main, dz_wrist], dim=1), zt_cat)   # real Δz → a
        acyc = self.h(torch.cat([gm, gw], dim=1), zt_cat)             # 왕복 (grad→g)
        l_recon = nn.functional.l1_loss(ahat, chunk)
        l_cycle = nn.functional.l1_loss(acyc, chunk)
        total = (w["align"] * (l_am + l_aw)
                 + w["recon"] * l_recon + w["cycle"] * l_cycle)
        return total, {"align_main": l_am.item(), "align_wrist": l_aw.item(),
                       "recon": l_recon.item(), "cycle": l_cycle.item()}
