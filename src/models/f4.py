"""C1 / F4 — 계층적 2채널 액션코드의 fine 채널 ζ_f (cowork §D, D1–D4).

액션코드 ζ = [ζ_g ⊕ ζ_f].
  ζ_g = 현행 pooled Δ (FlowPolicy가 그대로 생성 — policy.py 무변경, 언어정렬 공간).
  ζ_f = 이 모듈. K개 저차원 연속 병목 토큰(VQ 아님). SigLIP2-large-256 patch ΔF에서
        텍스트 초기화 K-쿼리 cross-attention으로 추출(D2).

설계 핵심 — 게이트 뒤 무영향(gated, zero-effect at init):
  • 인코더 tanh-gate α=0 (Flamingo, D2 붕괴 방어) → 초기 ζ_f=0.
  • 디코더 fine-head tanh-gate β=0 → 초기 액션 기여=0 → pooled-only와 비트 동형(bit-identical).
    (policy.py의 ζ_g flow는 손대지 않으므로 ζ̂_g·frozen 디코딩 경로가 정확히 보존됨.)
  • module.f4 없거나 disable이면 이 모듈 자체를 만들지 않음 → state_dict 키 불변, 기존 ckpt strict 로드.

ζ_f 생성(폐루프 대응): 학습 타깃 ζ_f_target = encode(ΔF, text). 추론 시 ΔF(미래 patch)는
  없으므로, base 토큰(z_prev,z_cur,g(A_past),lang,wrist) 조건의 작은 conditional flow가
  noise→ζ_f 로 생성(D3-A "ζ_f source=noise"). 폐루프 롤아웃은 이 flow_sample을 사용.

DEVIATION LEDGER (cowork §D 대비):
  D3-A 원문은 "[ζ_g;ζ_f] concat 단일 벡터에 공유 τ flow(신규 모듈 0)". 그러나 v_net을 넓히면
  RNG 소비·상태 결합으로 ζ̂_g가 초기부터 달라져 요구된 bit-identity(enabled@init=pooled 불변)를
  만족 못 함. → ζ_g flow는 무변경, ζ_f를 별도 gated branch로 분리 구현(공유 τ, noise source).
  수송 의미 동등. UNVERIFIED: cowork가 non-bit-identical 단일-vnet을 선호할 수 있음(검토 요망).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.policy import ResidualBlock


class F4FineChannel(nn.Module):
    def __init__(self, dense_dim, text_dim, latent_dim, n_base_tokens,
                 action_dim, n_chunk, n_patch=256, K=8, bottleneck_dim=96,
                 d_attn=768, heads=8, flow_steps=6, d_flow=512, flow_layers=3):
        super().__init__()
        self.K, self.bneck = K, bottleneck_dim
        self.n_chunk, self.action_dim = n_chunk, action_dim
        self.flow_steps = flow_steps
        self.zf_dim = K * bottleneck_dim

        # ---- ζ_f 인코더 (D2): 텍스트 초기화 K-쿼리 cross-attn over [ΔF + 2D pos-emb] ----
        self.kv_proj = nn.Linear(dense_dim, d_attn)
        # 2D 위치임베딩: 16×16 그리드 각 셀에 학습 임베딩(고정 격자이므로 patch당 1개 = 2D 인코딩과 동형)
        self.pos_emb = nn.Parameter(torch.zeros(n_patch, d_attn))
        self.ln_kv = nn.LayerNorm(d_attn)
        self.text_q = nn.Linear(text_dim, d_attn)          # 쿼리 초기화 = 텍스트 임베딩 사영(§0: pooled 자기질의 금지)
        self.query_offset = nn.Parameter(torch.zeros(K, d_attn))   # K개 쿼리 구분용 학습 오프셋
        self.attn = nn.MultiheadAttention(d_attn, heads, batch_first=True)
        self.bottleneck = nn.Linear(d_attn, bottleneck_dim)        # 저차원 연속 병목(D1)
        self.alpha = nn.Parameter(torch.zeros(1))                  # tanh-gate α=0 (Flamingo, D2)

        # ---- ζ_f 생성 flow (noise → ζ_f), base 토큰 조건 (D3-A source=noise) ----
        cin = n_base_tokens * latent_dim
        self.ctx = nn.Sequential(nn.LayerNorm(cin), nn.Linear(cin, d_flow),
                                 *[ResidualBlock(d_flow) for _ in range(2)])
        self.t_embed = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, 128))
        self.v_in = nn.Linear(self.zf_dim + d_flow + 128, d_flow)
        self.v_blocks = nn.Sequential(*[ResidualBlock(d_flow) for _ in range(flow_layers)])
        self.v_out = nn.Sequential(nn.LayerNorm(d_flow), nn.Linear(d_flow, self.zf_dim))
        self.register_buffer("zf_std", torch.ones(1))

        # ---- 디코더 fine-head (D4): frozen h(ζ_g) 위 gated 잔차, β=0 → bit-identity ----
        din = latent_dim + self.zf_dim + latent_dim                # [ζ_g ; ζ_f ; z_cur]
        self.fine = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, d_flow), nn.GELU(),
                                  nn.Linear(d_flow, n_chunk * action_dim))
        self.beta = nn.Parameter(torch.zeros(1))                   # bit-identity gate

        # ---- L_consistency (D4): ζ_f → pooled 변위 Δp (ζ_f를 실제 변위에 묶음) ----
        self.consist = nn.Linear(self.zf_dim, latent_dim)

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.query_offset, std=0.02)

    # ---- 학습 타깃: 미래 patch ΔF에서 ζ_f 인코딩 ----
    def encode(self, dF, text_emb):                # dF (B,P,dense_dim), text_emb (B,text_dim)
        kv = self.ln_kv(self.kv_proj(dF) + self.pos_emb.unsqueeze(0))
        q = self.text_q(text_emb).unsqueeze(1) + self.query_offset.unsqueeze(0)   # (B,K,d_attn)
        o, _ = self.attn(q, kv, kv)                # (B,K,d_attn)
        z = self.bottleneck(o)                     # (B,K,bneck)
        zf = torch.tanh(self.alpha) * z            # α=0 → 초기 0
        return zf.flatten(1)                       # (B, K*bneck)

    # ---- ζ_f 생성 flow ----
    def _v(self, x, ctx, t):
        return self.v_out(self.v_blocks(self.v_in(torch.cat([x, ctx, self.t_embed(t)], dim=1))))

    def flow_sample(self, base_flat, generator=None):
        ctx = self.ctx(base_flat)
        x = torch.randn((len(base_flat), self.zf_dim), device=base_flat.device,
                        generator=generator) * self.zf_std
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((len(x), 1), i * dt, device=x.device)
            x = x + self._v(x, ctx, t) * dt
        return x

    def flow_fm_and_sample(self, base_flat, target, generator=None):
        """CFM 손실 + ODE 샘플 ζ̂_f 동시 반환 (policy.FlowPolicy.fm_and_sample 동형)."""
        ctx = self.ctx(base_flat)
        x0 = torch.randn((len(base_flat), self.zf_dim), device=base_flat.device,
                         generator=generator) * self.zf_std
        t = torch.rand((len(x0), 1), device=x0.device, generator=generator)
        xt = (1 - t) * x0 + t * target
        l_fm = F.mse_loss(self._v(xt, ctx, t), target - x0)
        dt = 1.0 / self.flow_steps
        x = x0
        for i in range(self.flow_steps):
            tt = torch.full((len(x), 1), i * dt, device=x.device)
            x = x + self._v(x, ctx, tt) * dt
        return x, l_fm

    # ---- 디코더 fine 잔차 (gated) ----
    def fine_action(self, zeta_g, zeta_f, z_cur):
        r = self.fine(torch.cat([zeta_g, zeta_f, z_cur], dim=1))
        r = r.view(-1, self.n_chunk, self.action_dim)
        return torch.tanh(self.beta) * r           # β=0 → 초기 0

    def consistency(self, zeta_f, disp):           # disp = z_next - z_cur (pooled)
        return F.mse_loss(self.consist(zeta_f), disp)


def build_f4_from_cfg(f4, dense_dim, text_dim, latent_dim, n_base_tokens,
                      action_dim, n_chunk, n_patch):
    return F4FineChannel(
        dense_dim=dense_dim, text_dim=text_dim, latent_dim=latent_dim,
        n_base_tokens=n_base_tokens, action_dim=action_dim, n_chunk=n_chunk,
        n_patch=n_patch, K=f4.get("K", 8),
        bottleneck_dim=f4.get("bottleneck_dim", 96),
        d_attn=f4.get("d_attn", 768), heads=f4.get("heads", 8),
        flow_steps=f4.get("flow_steps", 6), d_flow=f4.get("d_flow", 512),
        flow_layers=f4.get("flow_layers", 3))
