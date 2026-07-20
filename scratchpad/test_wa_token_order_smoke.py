# W-A/W-B 토큰-스택 동형성 스모크 (VERIFY A2 게이트; CPU, 합성 텐서, 캐시/ckpt 불요).
#
# canonical 순서 (DESIGN_wrist_fusion_unified_v1 §3.2 [AMEND A2] 확정):
#   base(#1–5: z_prev, z_cur, a_emb, lang, wrist_sig) → grid(#6–9) → w_tok(#10, 항상 마지막)
#
# 검증 5단:
#   (1) train측(train_phase2.forward)과 rollout측(rollout_sim.py)의 토큰 조립을 그대로
#       전사해, 동일 합성 입력 + 공유 GridObs(eval)로 두 스택이 텐서 단위로 완전
#       동일(순서·값)함을 assert — **W-B w_tok(#10) 실배선 포함** (n_tokens=10).
#   (2) 각 토큰을 유일 상수로 채워 스택 행 순서 = canonical 순서임을 assert.
#   (3) W-B 시간 동형성: 학습측 인덱싱 w(t)=p̄(t)−p̄(max(t−span,0)) (libero.py obs_delta)
#       vs 롤아웃측 링버퍼(deque maxlen=span//H+1, append 후 [0] 읽기 — rollout_sim.py)를
#       합성 시계열에서 재계획 시점마다 대조 — 값 완전 동일 assert (과거 전용, 클램프 동형).
#   (4) N2 재척도 동형성: 학습(pre-scale numpy) vs 롤아웃(스칼라 곱 torch) 동일 값.
#   (5) 소스 텍스트 검사: 두 파일 모두 base → grid → w_tok → stack 순서 + w_tok 이
#       base 리스트에 끼어들지 않았음 + libero.py obs_delta 클램프 식 존재를 assert.
import sys
from pathlib import Path

import torch

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))
from models.obs_fusion import GridObs  # noqa: E402

torch.manual_seed(0)
torch.set_grad_enabled(False)              # 스모크: 순전파 비교만 (grad 불요)
B, D, Kg = 1, 1024, 4                      # W-A: latent 1024, wrist patch pool2 → P=4
use_lang = use_wrist = True
use_wdelta = True                          # W-B 실배선 검증

# 공유 모듈·입력 (train/rollout 동일 가중치·동일 프레임 가정)
grid_obs = GridObs(patch_dim=D, out_dim=D, n_tokens=Kg, pool="avg",
                   ln=True, tok_drop=0.1, group_drop=0.1, init_std=0.02).eval()
patches = torch.randn(B, 4, D)             # pool2 캐시 동형: P=2×2=4 (g_out=2 → 풀 항등)

# 유일 상수 토큰 (순서 뒤바뀜이 값으로 드러나도록)
zp_c, zc_c = torch.full((B, D), 1.0), torch.full((B, D), 2.0)
aemb, lang, wr_c = (torch.full((B, D), 3.0), torch.full((B, D), 4.0),
                    torch.full((B, D), 5.0))
wd = torch.full((B, D), 6.0)               # w_tok (N2 재척도 완료본으로 가정)

_pad = lambda t: t                          # W-A/B 단일-스트림: 전 토큰 1024 → pad no-op  # noqa: E731

# ── (1a) train측 전사: train_phase2.py forward (base → grid append → w_tok append → stack)
base = [_pad(zp_c), _pad(zc_c), aemb] + ([_pad(lang)] if use_lang else []) \
    + ([_pad(wr_c)] if use_wrist else [])
toks = list(base)
grid_tok = grid_obs(patches)                                   # (B, Kg, D)
toks = toks + [grid_tok[:, k] for k in range(Kg)]
if use_wdelta:                                                 # W-B w_tok — canonical 마지막
    toks = toks + [_pad(wd)]
train_stack = torch.stack(toks, dim=1)                         # (B, 10, D)

# ── (1b) rollout측 전사: rollout_sim.py (toks 리스트 → grid raw 공유 → w_tok append → pad → stack)
toks_r = [zp_c, zc_c, aemb] + ([lang] if use_lang else []) \
    + ([wr_c] if use_wrist else [])
raw = patches                                                  # grid_raw(obs) 동형
gt = grid_obs(raw)                                             # grid_toks(obs, raw=raw) 동형
toks_r = toks_r + [gt[:, k] for k in range(gt.size(1))]
toks_r = toks_r + [wd]                                         # w_tok #10 (마지막)
_zd = zp_c.shape[-1]
toks_r = [t if t.shape[-1] == _zd else
          torch.nn.functional.pad(t, (0, _zd - t.shape[-1])) for t in toks_r]
rollout_stack = torch.stack(toks_r, dim=1)

assert train_stack.shape == rollout_stack.shape == (B, 5 + Kg + 1, D), train_stack.shape
assert torch.equal(train_stack, rollout_stack), \
    "train/rollout 토큰 스택 불일치 (순서 또는 값)"

# ── (2) canonical 순서: base 5개 = 상수 1..5, 이어서 grid 4개(k=0..3), 마지막 w_tok=6
means = train_stack[0, :5].mean(dim=-1)
assert torch.allclose(means, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), means
for k in range(Kg):
    assert torch.equal(train_stack[0, 5 + k], grid_tok[0, k]), f"grid 토큰 {k} 순서 불일치"
assert train_stack.shape[1] == 10 and float(train_stack[0, -1].mean()) == 6.0, \
    "w_tok #10(마지막) 규약 위반"

# ── (3)+(4) W-B 시간 동형성 + N2 재척도: 학습 인덱싱 vs 롤아웃 링버퍼 (합성 시계열)
import collections  # noqa: E402
import numpy as np  # noqa: E402

span, H = 16, 8
T = 8 * H + 1
Dseq = np.random.RandomState(1).randn(T, 4, D).astype(np.float32)   # dense 캐시 동형
scale = 7.31                                                        # N2 σ_ref/σ_Δ (임의 양수)
wd_hist = collections.deque(maxlen=span // H + 1)                   # rollout_sim.py 전사
for t in range(0, T, H):                                            # 재계획 시점
    # 학습측 (libero.py build_policy_samples obs_delta + train_phase2 N2 pre-scale)
    w_train = (Dseq[t].mean(0) - Dseq[max(t - span, 0)].mean(0)) * scale
    # 롤아웃측 (rollout_sim.py: pmean append 후 [0] 읽기, 스칼라 곱)
    pmean = torch.tensor(Dseq[t]).mean(dim=0)
    wd_hist.append(pmean)
    w_roll = (pmean - wd_hist[0]) * scale
    assert torch.allclose(w_roll, torch.tensor(w_train), atol=1e-5), \
        f"t={t}: 링버퍼 w_tok ≠ 학습 인덱싱 w_tok (과거 클램프/캐던스 불일치)"
assert float(torch.tensor(
    (Dseq[0].mean(0) - Dseq[0].mean(0))).abs().max()) == 0.0        # t=0 클램프 → 0

# ── (5) 소스-순서 검사 (전사 드리프트 방호)
tp2 = (WS / "src/training/train_phase2.py").read_text()
rs = (WS / "src/eval_libero/rollout_sim.py").read_text()
lb = (WS / "src/data/libero.py").read_text()
fwd = tp2[tp2.index("def forward(zp, zc, zn, aemb, cf, lang, wr, *dobs"):]
assert fwd.index("base = [_pad(zp_c)") < fwd.index("[grid_tok[:, k] for k in range(Kg)]") \
    < fwd.index("toks = toks + [_pad(wd)]") \
    < fwd.index("toks = torch.stack(toks, dim=1)"), \
    "train: base→grid→w_tok→stack 순서 깨짐"
seg = rs[rs.index("toks = [zp_c, zc_c, a_emb]"):]
assert seg.index("toks = toks + grid_toks(obs, raw=raw)") \
    < seg.index("toks = toks + [w_tok]") < seg.index("torch.stack(toks"), \
    "rollout(W-B): grid→w_tok→stack 순서 깨짐"
assert seg.index("toks = toks + grid_toks(obs)") < seg.index("torch.stack(toks"), \
    "rollout: grid append 가 stack 앞이 아님"
assert "wrist_delta" not in fwd.split("base = [_pad(zp_c)")[1].split("toks = list(base)")[0], \
    "w_tok 이 base 리스트에 끼어듦 (canonical #10 위반)"
assert "D_first[t].mean(0) - D_first[max(t - self.span, 0)].mean(0)" in lb, \
    "libero.py obs_delta 클램프 식 부재/변형"
assert "wd_hist.append(pmean)" in rs and "pmean - wd_hist[0]" in rs, \
    "rollout_sim W-B 링버퍼 배선 부재/변형"

print("A2 SMOKE OK — train/rollout 토큰 스택 동형"
      f" (shape {tuple(train_stack.shape)}), canonical 순서 base(5)→grid(4)→w_tok(#10),"
      " W-B 링버퍼=학습 인덱싱 시간 동형(N2 재척도 포함) 확인")
