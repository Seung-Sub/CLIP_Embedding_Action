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


class GridObs(nn.Module):
    """F5-H L1 — UNGATED 그리드 관측 토큰 (cowork COWORK_DESIGN_F5H §1 L1, 선택 팔).

    DINOv3(등) dense patch 격자 (B, P=g×g, patch_dim) → **가벼운 공간 풀링**으로
    K=g_out×g_out 토큰(위치 보존, MolmoAct 선례) → 각 토큰을 정책 잠재폭(out_dim)으로
    사영 → phase2 flow 조건 토큰열 끝에 **UNGATED 상시 삽입**(no tanh gate, step0부터).

    ObsFusion(F3)과의 차이 — 왜 별도 모듈인가:
      • ObsFusion의 pool 은 'attn'(K개 학습쿼리 cross-attn = F3가 실패한 학습형 요약) 또는
        'mean'(전 patch 평균 → 1토큰, 공간 전멸)뿐 — **공간 격자를 보존하는 저해상 풀이 없음**.
      • GridObs는 patch 격자를 g_out×g_out 로 **avg-pool(파라미터-0)** 하여 위치/기하를 유지
        (S1b: 기하는 관측/조건 경로에서 도움). 학습 부품은 사영 1개(A1 "학습 최소화").
      • 단일 앵커(DINOv3) 전용 — 다중 인코더 융합(ObsFusion)과 목적/구조가 다름.
    ★ C1/C2(f4.py)와의 차이: C1/C2는 텍스트-쿼리 cross-attn dense patch → tanh(α/β) 게이트
      **타깃/액션코드 측** fine 잔차(게이트 미개방=굶음). GridObs는 게이트 없이 **관측/조건 측**
      에 상시 삽입 → cold-start 없음. (단 정직한 리스크: F3의 obs 토큰도 이미 ungated append였고
      폐루프가 richer-obs로 악화했음 — 아래 pool='avg'의 위치보존·저해상이 유일한 구조적 차이.)

    pool:
      'avg' / '2x2' : adaptive_avg_pool2d(g×g → g_out×g_out) — 파라미터-0 공간 다운샘플.
                      출력 격자는 n_tokens 로 결정(g_out=round(√n_tokens); 기본 16 → 4×4).
      'attn'        : K개 학습쿼리 × cross-attn(학습형 풀; ObsFusion.attn 단일앵커판 — ablation용).
    """
    def __init__(self, patch_dim, out_dim, n_tokens=16, pool="avg",
                 d_attn=768, heads=8):
        super().__init__()
        assert pool in ("avg", "2x2", "attn"), pool
        self.pool = "avg" if pool in ("avg", "2x2") else "attn"
        self.n_tokens = int(n_tokens)
        self.g_out = int(round(self.n_tokens ** 0.5))
        assert self.g_out * self.g_out == self.n_tokens, \
            f"n_tokens={n_tokens} 는 공간 격자용 완전제곱이어야 함 (예 16→4×4)"
        self.out_dim = out_dim
        if self.pool == "avg":
            self.proj = nn.Linear(patch_dim, out_dim)          # 학습 파라미터 = 사영 1개
        else:
            self.kv_proj = nn.Linear(patch_dim, d_attn)
            self.ln = nn.LayerNorm(d_attn)
            self.query = nn.Parameter(torch.randn(n_tokens, d_attn))
            self.attn = nn.MultiheadAttention(d_attn, heads, batch_first=True)
            self.proj = nn.Linear(d_attn, out_dim)

    def forward(self, patches):                            # (B, P=g×g, patch_dim)
        import torch.nn.functional as F
        B, P, D = patches.shape
        if self.pool == "avg":
            g = int(round(P ** 0.5))
            assert g * g == P, f"patch 격자 P={P} 가 정사각 아님 (avg-pool 불가)"
            x = patches.reshape(B, g, g, D).permute(0, 3, 1, 2)          # (B,D,g,g)
            x = F.adaptive_avg_pool2d(x, (self.g_out, self.g_out))       # (B,D,go,go)
            x = x.permute(0, 2, 3, 1).reshape(B, self.n_tokens, D)       # (B,K,D)
            return self.proj(x)                                          # (B,K,out_dim)
        q = self.query.unsqueeze(0).expand(B, -1, -1)                    # (B,K,d_attn)
        kv = self.ln(self.kv_proj(patches))
        o, _ = self.attn(q, kv, kv)
        return self.proj(o)                                              # (B,K,out_dim)
