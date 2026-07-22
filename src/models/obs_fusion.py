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

    W-A(WristCond-v1) guarded 옵션 (DESIGN_wrist_fusion_unified_v1 §2.3 N1 · VERIFY A1;
    기본값은 전부 off = 기존 config 비트/바이트 동형 — 모듈 생성·RNG 소비·state_dict 키 불변):
      ln=True       : 사영(proj) **앞** LayerNorm(patch_dim) — N1, F3 kv-LN 결함의 param-최소 대응.
                      attn 풀에선 kv_proj 앞에 적용. False(기본)면 서브모듈 자체를 만들지 않음.
      tok_drop      : 학습시만, 사영 후 토큰별 독립 zero-drop p (rescale 없음 — zero-ablation 동형).
      group_drop    : 학습시만, K개 그리드 토큰 **전부를 함께** 0으로 (패치 문서 §2.3 ②:
                      "기하 토큰 없이도 동작하는 언어-only 경로" 유지 강제 — modality-dropout 관례).
      init_std      : proj weight 를 N(0, init_std²)·bias 0 으로 재초기화 (설계 규격 0.02).
                      None(기본)이면 torch 기본 init 그대로 (RNG 스트림 불변).
    """
    def __init__(self, patch_dim, out_dim, n_tokens=16, pool="avg",
                 d_attn=768, heads=8, ln=False, tok_drop=0.0, group_drop=0.0,
                 init_std=None):
        super().__init__()
        assert pool in ("avg", "2x2", "attn"), pool
        self.pool = "avg" if pool in ("avg", "2x2") else "attn"
        self.n_tokens = int(n_tokens)
        self.g_out = int(round(self.n_tokens ** 0.5))
        assert self.g_out * self.g_out == self.n_tokens, \
            f"n_tokens={n_tokens} 는 공간 격자용 완전제곱이어야 함 (예 16→4×4)"
        self.out_dim = out_dim
        self.tok_drop = float(tok_drop)
        self.group_drop = float(group_drop)
        assert 0.0 <= self.tok_drop < 1.0 and 0.0 <= self.group_drop < 1.0, \
            (tok_drop, group_drop)
        # ln=False(기본)면 미생성 → 기존 ckpt 와 state_dict 키 동일 (byte-identity)
        self.in_ln = nn.LayerNorm(patch_dim) if ln else None
        if self.pool == "avg":
            self.proj = nn.Linear(patch_dim, out_dim)          # 학습 파라미터 = 사영 1개
        else:
            self.kv_proj = nn.Linear(patch_dim, d_attn)
            self.ln = nn.LayerNorm(d_attn)
            self.query = nn.Parameter(torch.randn(n_tokens, d_attn))
            self.attn = nn.MultiheadAttention(d_attn, heads, batch_first=True)
            self.proj = nn.Linear(d_attn, out_dim)
        if init_std is not None:                               # 설계 규격 std 0.02 (guarded)
            nn.init.normal_(self.proj.weight, std=float(init_std))
            nn.init.zeros_(self.proj.bias)

    def _drop(self, x):                                    # (B, K, out_dim) — 학습시만 활성
        if not self.training or (self.tok_drop <= 0 and self.group_drop <= 0):
            return x                                       # eval/off = 항등 (비트 동형)
        B = x.shape[0]
        if self.tok_drop > 0:                              # ① per-token zero-drop (독립)
            keep = (torch.rand(B, self.n_tokens, 1, device=x.device) >= self.tok_drop)
            x = x * keep.to(x.dtype)
        if self.group_drop > 0:                            # ② group-drop: K개 전부 0 (언어-only 경로)
            keep = (torch.rand(B, 1, 1, device=x.device) >= self.group_drop)
            x = x * keep.to(x.dtype)
        return x

    def forward(self, patches):                            # (B, P=g×g, patch_dim)
        import torch.nn.functional as F
        B, P, D = patches.shape
        if self.pool == "avg":
            g = int(round(P ** 0.5))
            assert g * g == P, f"patch 격자 P={P} 가 정사각 아님 (avg-pool 불가)"
            x = patches.reshape(B, g, g, D).permute(0, 3, 1, 2)          # (B,D,g,g)
            x = F.adaptive_avg_pool2d(x, (self.g_out, self.g_out))       # (B,D,go,go)
            x = x.permute(0, 2, 3, 1).reshape(B, self.n_tokens, D)       # (B,K,D)
            if self.in_ln is not None:                                   # N1: 사영 전 LN(patch_dim)
                x = self.in_ln(x)
            return self._drop(self.proj(x))                              # (B,K,out_dim)
        q = self.query.unsqueeze(0).expand(B, -1, -1)                    # (B,K,d_attn)
        kv = patches if self.in_ln is None else self.in_ln(patches)      # N1 (attn 판: kv_proj 앞)
        kv = self.ln(self.kv_proj(kv))
        o, _ = self.attn(q, kv, kv)
        return self._drop(self.proj(o))                                  # (B,K,out_dim)


class LangSelPool(nn.Module):
    """P-B — 텍스트-조건 patch 풀링 (OTTER-style, DESIGN_patch_policy_attention_v1 §4).

    지시문(lang) 임베딩이 dense patch 격자를 **질의**해 "이 태스크에 유관한 기하"
    K개 토큰을 추출, phase2 flow 조건 토큰열 끝에 **UNGATED 상시 삽입**한다(IV4).
    C1/C2(f4)와 같은 텍스트-쿼리 cross-attn 연산이지만 삽입점이 다르다 — 타깃/코드
    측(gated, ∂L_act/∂α≡0)이 아니라 **관측/조건화 측**(태스크 손실 직통, 게이트 없음).

    F3 결함-수정 체크리스트(설계 §2.3) 전항 반영:
      • kv-LN: kv = ln_kv(kv_proj(patch) + pos_emb) — f4.py _readout 배선 동형(사내 검증).
      • init: 모든 학습 Linear weight N(0, 0.02²)·bias 0, pos_emb/query_offset std 0.02
        (F3 의 query randn std 1.0 결함 수리; MHA 내부는 torch 기본 = f4 동형).
      • pos-emb: kv측 학습 2D 격자 임베딩 (P, d_attn) — attention 풀링은 위치 필수.
      • dropout 2단(학습시만): per-token zero-drop + group-drop(K개 전부 0 — 언어-only
        경로 보존 강제, modality-dropout 관례). GridObs._drop 규약 동일.
    쿼리 = W_q·L2norm(lang) + query_offset_k — lang L2-norm 은 §4.4 스케일-지배 방지.
    출력 = out(ln_out(attn_out)) (B, K, out_dim) — 정책 잠재폭 사영.

    forward(..., return_attn=True) 는 (tokens, attn_weights(B,heads,K,P)) 반환 —
    설계 §4.2 필수 로깅(attention entropy·‖token‖)용. 기본 False 는 tokens만(빠름).
    이 클래스 추가는 기존 모듈(ObsFusion/GridObs)의 코드·RNG 소비·state_dict 에 어떤
    영향도 없음 (byte-identity: patch_obs 미설정 경로는 이 클래스를 생성하지 않는다).
    """

    def __init__(self, patch_dim, text_dim, out_dim, n_patch, n_tokens=1,
                 d_attn=768, heads=8, tok_drop=0.1, group_drop=0.1):
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.n_patch = int(n_patch)
        self.out_dim = out_dim
        self.tok_drop = float(tok_drop)
        self.group_drop = float(group_drop)
        assert 0.0 <= self.tok_drop < 1.0 and 0.0 <= self.group_drop < 1.0, \
            (tok_drop, group_drop)
        self.kv_proj = nn.Linear(patch_dim, d_attn)
        self.pos_emb = nn.Parameter(torch.zeros(self.n_patch, d_attn))
        self.ln_kv = nn.LayerNorm(d_attn)
        self.text_q = nn.Linear(text_dim, d_attn)
        self.query_offset = nn.Parameter(torch.zeros(self.n_tokens, d_attn))
        self.attn = nn.MultiheadAttention(d_attn, heads, batch_first=True)
        self.ln_out = nn.LayerNorm(d_attn)
        self.out = nn.Linear(d_attn, out_dim)
        # 설계 규격 init (§2.3): Linear weight std 0.02 / bias 0, 임베딩 std 0.02.
        for lin in (self.kv_proj, self.text_q, self.out):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.query_offset, std=0.02)

    def _drop(self, x):                                    # (B, K, out_dim) — 학습시만 활성
        if not self.training or (self.tok_drop <= 0 and self.group_drop <= 0):
            return x                                       # eval/off = 항등
        B = x.shape[0]
        if self.tok_drop > 0:                              # ① per-token zero-drop (독립)
            keep = (torch.rand(B, self.n_tokens, 1, device=x.device) >= self.tok_drop)
            x = x * keep.to(x.dtype)
        if self.group_drop > 0:                            # ② group-drop: K개 전부 0 (언어-only 경로)
            keep = (torch.rand(B, 1, 1, device=x.device) >= self.group_drop)
            x = x * keep.to(x.dtype)
        return x

    def forward(self, patches, lang, return_attn=False):
        """patches (B, P, patch_dim) + lang (B, text_dim) → (B, K, out_dim).

        return_attn=True 면 (tokens, weights(B, heads, K, P)) — entropy/맵 로깅용."""
        import torch.nn.functional as F
        B, P, _D = patches.shape
        assert P == self.n_patch, \
            f"patch 수 {P} != pos_emb {self.n_patch} (pool_to/캐시 키 불일치?)"
        kv = self.ln_kv(self.kv_proj(patches) + self.pos_emb.unsqueeze(0))
        q = self.text_q(F.normalize(lang, dim=-1)).unsqueeze(1) \
            + self.query_offset.unsqueeze(0)               # (B, K, d_attn)
        o, w = self.attn(q, kv, kv, need_weights=return_attn,
                         average_attn_weights=False)
        tok = self._drop(self.out(self.ln_out(o)))         # (B, K, out_dim) — UNGATED
        return (tok, w) if return_attn else tok
