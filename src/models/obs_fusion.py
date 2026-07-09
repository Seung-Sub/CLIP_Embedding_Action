"""F3 — 다중 시각 인코더 관측 융합 (obs fusion): patch 토큰 → 정책 잠재 토큰.

여러 사전학습 인코더(예: DINOv2-reg 1024, SigLIP2 1152)의 patch 임베딩을
공통 폭 d_attn으로 사영한 뒤 하나의 토큰열로 결합, 학습형 query가
cross-attention으로 n_query개의 요약 토큰(ζ̂ 후보)을 뽑는다.

pool 2종 (공통 출력 (B, Q, out_dim); attn: Q=n_query, mean: Q=1):
  attn : 학습 query × cross-attn 풀링 (권장 — 인코더별 정보 선택적 결합)
  mean : query/attn 없이 인코더별 patch 평균 → 사영 → 인코더축 합산 (베이스라인)

out_dim 은 정책 latent_dim (CLIP=768)과 일치해야 함 — 호출측에서 보장할 것
(하드 실패 대신 문서화; build 시 assert 로 유도).

pixel-unshuffle(unshuffle>1): 정사각 patch 격자(P=g×g)에서 u×u 이웃을 묶어
토큰수 1/u², 폭 ×u² 로 재배열(공간 다운샘플). 이때 proj 는 넓어진 폭
(dim·u²)을 입력으로 받도록 구성한다. 격자가 정사각이 아니거나 g가 u로
나눠지지 않으면 unshuffle을 생략(경고)하고 폭만 0패딩해 차원을 맞춘다.
"""
import warnings

import torch
import torch.nn as nn


class ObsFusion(nn.Module):
    def __init__(self, encoder_dims, d_attn=768, n_query=8, out_dim=768,
                 heads=8, unshuffle=1, pool="attn"):
        super().__init__()
        assert pool in ("attn", "mean"), pool
        self.pool = pool
        self.unshuffle = unshuffle
        self.n_query = n_query if pool == "attn" else 1
        self.out_dim = out_dim
        # unshuffle>1 이면 proj 입력폭 = dim·u² (넓어진 폭을 proj가 흡수)
        u2 = unshuffle * unshuffle
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim * u2 if unshuffle > 1 else dim, d_attn)
            for name, dim in encoder_dims.items()})
        self.ln = nn.LayerNorm(d_attn)
        self.out = nn.Linear(d_attn, out_dim)
        if pool == "attn":
            self.query = nn.Parameter(torch.randn(n_query, d_attn))
            self.attn = nn.MultiheadAttention(d_attn, heads, batch_first=True)

    def _pixel_unshuffle(self, x, u):
        """(B, P, dim) → (B, P/u², dim·u²). P=g×g 정사각·g%u==0 가정.
        불충족 시 None 반환(호출측이 0패딩 폴백)."""
        B, P, dim = x.shape
        g = int(round(P ** 0.5))
        if g * g != P or g % u != 0:
            warnings.warn(
                f"unshuffle={u}: patch 격자 P={P}가 정사각/가분(g%u==0) 아님 "
                f"→ unshuffle 생략, 0패딩 폴백", stacklevel=2)
            return None
        x = x.view(B, g, g, dim)
        x = x.view(B, g // u, u, g // u, u, dim)         # 행/열을 u단위 블록으로
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()     # (B, g/u, g/u, u, u, dim)
        return x.view(B, (g // u) ** 2, dim * u * u)      # 이웃 u² 를 폭으로 접음

    def forward(self, patch_dict):                       # {name: (B, P_e, dim_e)}
        toks = {}
        for name, x in patch_dict.items():
            if self.unshuffle > 1:
                xu = self._pixel_unshuffle(x, self.unshuffle)
                if xu is None:                           # 폴백: 토큰수 유지, 폭만 0패딩
                    B, P, dim = x.shape
                    pad = x.new_zeros(B, P, dim * (self.unshuffle ** 2 - 1))
                    x = torch.cat([x, pad], dim=-1)
                else:
                    x = xu
            toks[name] = self.proj[name](x)              # (B, P'_e, d_attn)

        if self.pool == "mean":                          # 인코더별 patch평균 → 합산
            pooled = sum(t.mean(dim=1) for t in toks.values())   # (B, d_attn)
            return self.out(self.ln(pooled)).unsqueeze(1)        # (B, 1, out_dim)

        kv = torch.cat(list(toks.values()), dim=1)       # (B, ΣP'_e, d_attn)
        q = self.query.unsqueeze(0).expand(kv.size(0), -1, -1)   # (B, n_query, d_attn)
        o, _ = self.attn(q, kv, kv)                      # cross-attn 풀링
        return self.out(self.ln(o))                      # (B, n_query, out_dim)
