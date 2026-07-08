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
                 decoder_state_cond=None, encoder_state_cond=None):
        """decoder_state_cond/encoder_state_cond: h/g 각각 독립적으로 상태조건을
        끄기 위한 오버라이드. None이면 state_cond와 동일(기존 동작 유지).
          decoder_state_cond=False (C0) : h가 z_t 없이도 되는지 (실측: 거의 무손실)
          encoder_state_cond=False (C1) : g가 z_t 없이 액션만으로 Δz를 예측 가능한지
                                          — "행동만으로 이미지변화가 결정되는가" 검증"""
        super().__init__()
        enc_cond = state_cond if encoder_state_cond is None else encoder_state_cond
        dec_cond = state_cond if decoder_state_cond is None else decoder_state_cond
        self.g = ChunkEncoder(action_dim, latent_dim, hidden, layers,
                              dropout, enc_cond)
        self.h = ChunkDecoder(action_dim, n_chunk, latent_dim, hidden,
                              layers, dropout, dec_cond)

    def losses(self, chunk, delta_z, w, z_t=None, align_type="mse_cos"):
        """align_type: mse_cos(기본, MSE+0.5·(1−cos)) | l1(순수 L1 — V-JEPA류 월드모델
        문헌에서 가장 흔한 선택, 방향/크기를 분리하지 않고 원시벡터 거리 하나로 처리)."""
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
        total = w["align"] * l_align + w["recon"] * l_recon + w["cycle"] * l_cycle
        return total, {"align": l_align.item(), "recon": l_recon.item(),
                       "cycle": l_cycle.item()}
