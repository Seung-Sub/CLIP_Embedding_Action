"""잠재 정책 f: 토큰 [z_{t−n}, z_t, g(A_past), (lang), (wrist)] → ζ̂ (768).

모듈 2종 (공통 인터페이스: forward(tokens (B,N,768)) -> (B,768)):
  - MLPConcat  : 통짜 결합 MLP 회귀 (베이스라인)
  - FlowPolicy : 조건부 flow matching (권장 — 캠페인 승자)

회귀 손실 (policy_losses):
  L = λ_lat·[MSE+0.5(1−cos)](ζ̂, g(A_fut, z_t))   # 주 잠재 GT (VITA L_FM 자리)
    + λ_act·L1(h(ζ̂, z_t), A_fut)                  # action 손실 (FLD 대응)
    + λ_wm ·0.5(1−cos)(ζ̂, z_next − z_t)          # 보조 (기각됨, 가중치 0)
flow 손실은 train_phase2의 flow 분기 참조 (CFM + FLD).
"""
import torch
import torch.nn as nn


class MLPConcat(nn.Module):
    def __init__(self, d_model=512, layers=4, heads=None, n_tokens=3, latent_dim=768):
        super().__init__()
        dims = [n_tokens * latent_dim] + [d_model] * (layers - 1)
        net = []
        for i in range(len(dims) - 1):
            net += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        net.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(nn.LayerNorm(n_tokens * latent_dim), *net)

    def forward(self, tokens):                    # (B, 3, 768)
        return self.net(tokens.flatten(1))


class ResidualBlock(nn.Module):
    """pre-LN 잔차 FFN 블록 (트랜스포머 FFN 동형) — 순수 MLP의 깊이 포화 해소.

    dropout=0.0(기본)이면 Dropout 모듈을 만들지 않아 Sequential 인덱스가 그대로
    → state_dict 키가 기존과 완전 동일(비트 동형, 기존 체크포인트 로드 유지)."""

    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        ff = [nn.Linear(d, 4 * d), nn.GELU()]
        if dropout > 0:
            ff.append(nn.Dropout(dropout))
        ff.append(nn.Linear(4 * d, d))
        self.ff = nn.Sequential(*ff)

    def forward(self, x):
        return x + self.ff(self.ln(x))


class FlowPolicy(nn.Module):
    """조건부 flow matching 헤드 — ζ 공간 속도장 v(x, t | ctx), Euler K스텝 적분.

    source(수송 출발점) 3종 — 각각 다른 문헌의 결합(coupling):
      noise  : x0 ~ N(0, x0_std²)      (π0 / Diffusion Policy 계열)
      past   : x0 = g(A_past) 토큰      (A2A식 액션→액션, 시간 연속성 활용)
      vision : x0 = z_cur 토큰          (VITA식 시각→액션 수송)
    x0_std 버퍼는 학습 시작 시 잠재 타깃 g(A_fut) 표준편차로 설정(체크포인트 저장).
    """

    A_EMB_IDX, Z_CUR_IDX = 2, 1                   # 토큰 위치 규약 고정

    def __init__(self, d_model=1024, layers=4, heads=None, n_tokens=3,
                 steps=6, source="past", ctx_layers=2, source_noise=0.1,
                 latent_dim=768, dropout=0.0,
                 flow_space="latent", action_flat_dim=None):
        super().__init__()
        assert source in ("noise", "past", "vision")
        self.steps, self.source, self.source_noise = steps, source, source_noise
        self.latent_dim = latent_dim
        # ★SWAP (action-space flow): flow_space='action'이면 CFM이 잠재 ζ 대신 RAW 액션청크
        # (flatten n_chunk*act_dim)를 직접 수송한다. ctx(조건) 토큰은 그대로 잠재(latent_dim)라
        # ctx-MLP 입력차원은 불변 — 오직 흐름 벡터 x/속도장 v_in·v_out의 차원(flow_dim)만 바뀜.
        # flow_space='latent'(기본)이면 flow_dim=latent_dim → v_in/v_out shape·키 완전 동일(regression-0).
        self.flow_space = flow_space
        if flow_space == "action":
            assert action_flat_dim is not None, \
                "flow_space='action'는 action_flat_dim(n_chunk*act_dim) 필요"
            assert source in ("past", "noise"), \
                "action flow는 source past|noise만 지원(vision 토큰은 잠재 스케일)"
            self.flow_dim = int(action_flat_dim)
        else:
            self.flow_dim = latent_dim
        # dropout(기본 0.0)은 ctx(=projector)·v_net 잔차블록 FFN에 적용. 0.0이면
        # ResidualBlock이 Dropout을 만들지 않아 state_dict 키가 기존과 동일(비트 동형).
        self.ctx = nn.Sequential(
            nn.LayerNorm(n_tokens * latent_dim),
            nn.Linear(n_tokens * latent_dim, d_model),
            *[ResidualBlock(d_model, dropout) for _ in range(ctx_layers)])
        self.t_embed = nn.Sequential(nn.Linear(1, 128), nn.GELU(),
                                     nn.Linear(128, 128))
        # flow_dim = latent_dim(기본) 또는 액션청크 flatten차원(SWAP). ctx(조건)는 항상 latent.
        self.v_in = nn.Linear(self.flow_dim + d_model + 128, d_model)
        self.v_blocks = nn.Sequential(*[ResidualBlock(d_model, dropout)
                                        for _ in range(layers)])
        self.v_out = nn.Sequential(nn.LayerNorm(d_model),
                                   nn.Linear(d_model, self.flow_dim))
        self.register_buffer("x0_std", torch.ones(1))

    def _v(self, x, ctx, t):
        h = self.v_in(torch.cat([x, ctx, self.t_embed(t)], dim=1))
        return self.v_out(self.v_blocks(h))

    def _x0(self, tokens, generator=None, x0_src=None):
        flow_dim = getattr(self, "flow_dim", self.latent_dim)
        if self.source == "noise":
            return torch.randn((len(tokens), flow_dim), device=tokens.device,
                               generator=generator) * self.x0_std
        # ★SWAP: action-space flow의 source='past' → x0 = 과거 정규화 액션청크(flatten) (외부 주입).
        # 잠재 flow(기본)은 종전대로 g(A_past)/z_cur 토큰을 x0 source로 사용(regression-0).
        if getattr(self, "flow_space", "latent") == "action":
            assert x0_src is not None, \
                "action flow source='past'는 x0_src(flatten 과거 액션청크) 필요"
            x0 = x0_src.clone()
        else:
            x0 = tokens[:, self.A_EMB_IDX if self.source == "past"
                        else self.Z_CUR_IDX].clone()
        if self.training and self.source_noise > 0:
            x0 = x0 + torch.randn(x0.shape, device=x0.device,
                                  generator=generator) \
                * (self.source_noise * self.x0_std)
        return x0

    def _integrate(self, x, ctx):
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((len(x), 1), i * dt, device=x.device)
            x = x + self._v(x, ctx, t) * dt
        return x

    def forward(self, tokens, generator=None, x0_src=None):    # 샘플링 (평가·롤아웃 공용)
        return self._integrate(self._x0(tokens, generator, x0_src),
                               self.ctx(tokens.flatten(1)))

    def fm_and_sample(self, tokens, target, generator=None, x0_src=None):
        """학습용: CFM 손실 + FLD용 ODE 샘플 ζ̂ (그래디언트 유지) 동시 반환.

        x0_src: action-space flow(source='past')일 때 flatten 과거 액션청크(외부 주입). 잠재 flow면 None."""
        ctx = self.ctx(tokens.flatten(1))
        x0 = self._x0(tokens, generator, x0_src)
        t = torch.rand((len(x0), 1), device=x0.device, generator=generator)
        xt = (1 - t) * x0 + t * target
        l_fm = nn.functional.mse_loss(self._v(xt, ctx, t), target - x0)
        return self._integrate(x0, ctx), l_fm


MODULES = {"mlp": MLPConcat, "flow": FlowPolicy}


def build_policy_from_cfg(m, n_tokens=3, latent_dim=768, action_flat_dim=None):
    """module 설정 dict → 정책 (flow 전용 키 포함). 학습·평가 공용 진입점.

    latent_dim = 앵커/DeltaAE 잠재 차원 (phase1 체크포인트에서 주입; CLIP=768).
    action_flat_dim = n_chunk*act_dim. module.flow_space='action'(SWAP)일 때만 필요(그 외 None).
    """
    kw = dict(d_model=m.get("d_model", 512), layers=m.get("layers", 4),
              heads=m.get("heads", 8), n_tokens=n_tokens, latent_dim=latent_dim)
    if m["name"] == "flow":
        kw.update(steps=m.get("flow_steps", 6),
                  source=m.get("flow_source", "past"),
                  ctx_layers=m.get("ctx_layers", 2),
                  source_noise=m.get("source_noise", 0.1),
                  dropout=m.get("dropout", 0.0),     # 기본 0.0 = 기존과 비트 동형
                  flow_space=m.get("flow_space", "latent"),   # ★SWAP: 'action'=액션청크 직접 flow (기본 latent=regression-0)
                  action_flat_dim=action_flat_dim)
    return MODULES[m["name"]](**kw)


def policy_losses(zeta, chunk_fut, z_cur, z_next, ae, w):
    """3항 손실. ae = 동결된 DeltaAE (g, h). chunk_fut (B,T,D) 정규화됨."""
    with torch.no_grad():
        lat_target = ae.g(chunk_fut, z_cur)               # 주 GT (동결 g)
    wm_target = z_next - z_cur
    cos = nn.functional.cosine_similarity
    l_lat = (nn.functional.mse_loss(zeta, lat_target)
             + 0.5 * (1 - cos(zeta, lat_target, dim=1)).mean())
    ahat = ae.h(zeta, z_cur)                              # 동결 디코딩 경로
    l_act = nn.functional.l1_loss(ahat, chunk_fut)
    l_wm = 0.5 * (1 - cos(zeta, wm_target, dim=1)).mean()
    total = w["lat"] * l_lat + w["act"] * l_act + w["wm"] * l_wm
    return total, {"lat": l_lat.item(), "act": l_act.item(), "wm": l_wm.item()}
