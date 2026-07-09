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


class DeltaAE(nn.Module):
    def __init__(self, action_dim, n_chunk, latent_dim=768, hidden=512,
                 layers=4, dropout=0.0, state_cond=True,
                 decoder_state_cond=None, encoder_state_cond=None,
                 align_mode="dz", contrast_w=0.0, contrast_loss="infonce",
                 contrast_head=False, sigmoid_bias0=-5.5):
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
        self.h = ChunkDecoder(action_dim, n_chunk, latent_dim, hidden,
                              layers, dropout, dec_cond)
        assert align_mode in ("dz", "direct", "hybrid"), align_mode
        self.align_mode = align_mode
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
        ahat = self.h(delta_z, z_t)              # 실제 Δz(+상태) → 액션 (FLD 대응)
        acyc = self.h(ghat, z_t)                 # 왕복 (phase2 디코딩 경로)
        if align_type == "l1":
            l_align = nn.functional.l1_loss(ghat, delta_z)
        else:
            cos = nn.functional.cosine_similarity(ghat, delta_z, dim=1)
            l_align = (nn.functional.mse_loss(ghat, delta_z)
                       + 0.5 * (1 - cos).mean())
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
        return total, parts
