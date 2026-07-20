# W-C(N3/N4) 스모크 + 기본 경로 byte-identity 검증 (CPU, 합성 텐서, ckpt/캐시 불요).
#
# 검증 4단:
#   (A) byte-identity vs HEAD: git HEAD 시점의 networks.py/policy.py 를 그대로 로드해
#       (1) DualDeltaAE(플래그 부재) — 동일 시드 생성 시 state_dict 완전 동일(키+값)
#           + losses() 출력 완전 동일, (2) FlowPolicy(x0_per_dim 부재) — state_dict
#           완전 동일 + eval forward 출력 완전 동일 을 assert.
#   (B) N3: 6.5× 스케일 불균형 합성 Δz 에서 stream_standardize=true 가 align 타깃
#       (표준화 공간)의 스트림별 std 를 각각 ≈1 로 정렬(비교 가능 크기)함을 assert
#       — off 면 6.5× 그대로. loss_terms 의 align_main/align_wrist 도 동일 크기級.
#   (C) 왕복(round-trip): std_dz → σ 곱 = 원본 (exact); ζ 통화 일관성 —
#       decode(std_dz(Δz)) == h([Δz/σ], z) 수동 계산과 동일, encode→decode 무스케일
#       통과(표준화 공간 왕복), phase1-eval 전사 경로 실행.
#   (D) N4: x0_per_dim=true 시 x0_std shape=(flow_dim,), per-dim std 주입 후 노이즈
#       x0 의 블록별 std 비율이 타깃 비율(6.5×)을 보존함을 assert (스칼라판은 ≈1).
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))
SCRATCH = Path(__file__).resolve().parent

torch.manual_seed(0)


def _load_head(rel, name):
    """git HEAD 시점 소스를 파일로 떠서 독립 모듈로 로드 (작업트리와 무간섭)."""
    src = subprocess.run(["git", "-C", str(WS), "show", f"HEAD:{rel}"],
                         capture_output=True, text=True, check=True).stdout
    p = SCRATCH / f"_head_{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(f"head_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


head_net = _load_head("src/models/networks.py", "networks")
head_pol = _load_head("src/models/policy.py", "policy")
from models.networks import DualDeltaAE  # noqa: E402  (작업트리 신판)
from models.policy import FlowPolicy  # noqa: E402

DM, DW, ACT, NC, B = 64, 48, 7, 16, 32
w = {"align": 0.5, "recon": 0.5, "cycle": 0.25}

# ── (A1) DualDeltaAE 플래그 부재 = HEAD 와 비트 동형 ─────────────────────────
torch.manual_seed(123)
ae_head = head_net.DualDeltaAE(ACT, NC, DM, DW, hidden=64, layers=2)
torch.manual_seed(123)
ae_new = DualDeltaAE(ACT, NC, DM, DW, hidden=64, layers=2)   # stream_standardize 미지정
sd_h, sd_n = ae_head.state_dict(), ae_new.state_dict()
assert list(sd_h.keys()) == list(sd_n.keys()), "state_dict 키 변형 (byte-identity 위반)"
assert all(torch.equal(sd_h[k], sd_n[k]) for k in sd_h), "state_dict 값 상이"
g = torch.Generator().manual_seed(7)
chunk = torch.randn(B, NC, ACT, generator=g)
dzm = torch.randn(B, DM, generator=g) * 0.1346
dzw = torch.randn(B, DW, generator=g) * 0.0208               # 6.5× 불균형 (BRIEF #7)
zm, zw = torch.randn(B, DM, generator=g), torch.randn(B, DW, generator=g)
ae_head.eval(); ae_new.eval()
with torch.no_grad():
    th, ph = ae_head.losses(chunk, dzm, dzw, zm, zw, w)
    tn, pn = ae_new.losses(chunk, dzm, dzw, zm, zw, w)
assert torch.equal(th, tn) and ph == pn, "플래그 부재 losses() 출력 상이 (비트 동형 위반)"
with torch.no_grad():
    assert torch.equal(ae_head.encode(chunk, zm, zw), ae_new.encode(chunk, zm, zw))
    assert torch.equal(ae_head.decode(torch.cat([dzm, dzw], 1), zm, zw),
                       ae_new.decode(torch.cat([dzm, dzw], 1), zm, zw))
print("(A1) DualDeltaAE flag-off == HEAD: state_dict/losses/encode/decode 비트 동형 OK")

# ── (A2) FlowPolicy x0_per_dim 부재 = HEAD 와 비트 동형 ──────────────────────
torch.manual_seed(321)
fp_head = head_pol.FlowPolicy(d_model=128, layers=2, n_tokens=6, latent_dim=DM + DW)
torch.manual_seed(321)
fp_new = FlowPolicy(d_model=128, layers=2, n_tokens=6, latent_dim=DM + DW)
sd_h, sd_n = fp_head.state_dict(), fp_new.state_dict()
assert list(sd_h.keys()) == list(sd_n.keys()) and \
    all(torch.equal(sd_h[k], sd_n[k]) for k in sd_h), "FlowPolicy state_dict 변형"
assert fp_new.x0_std.shape == torch.Size([1]), "기본 x0_std 가 스칼라(1,)가 아님"
toks = torch.randn(B, 6, DM + DW, generator=torch.Generator().manual_seed(9))
fp_head.eval(); fp_new.eval()                # eval: source_noise 미적용 → 결정론
with torch.no_grad():
    assert torch.equal(fp_head(toks), fp_new(toks)), "FlowPolicy forward 출력 상이"
print("(A2) FlowPolicy flag-off == HEAD: state_dict/forward 비트 동형 OK")

# ── (B) N3: 표준화가 6.5× 불균형을 비교 가능 크기로 정렬 ─────────────────────
torch.manual_seed(55)
ae_std = DualDeltaAE(ACT, NC, DM, DW, hidden=64, layers=2, stream_standardize=True)
ae_std.dz_std_main.fill_(dzm.std())          # train_phase1 주입 전사
ae_std.dz_std_wrist.fill_(dzw.std())
sm, sw = ae_std.std_dz(dzm, dzw)
r_raw = (dzm.std() / dzw.std()).item()
r_std = (sm.std() / sw.std()).item()
assert r_raw > 5.0, f"합성 불균형 확인 실패 ({r_raw:.2f})"
assert abs(r_std - 1.0) < 0.05, f"N3 후 스트림 std 비율 {r_std:.3f} ≠ ≈1"
terms = ae_std.loss_terms(chunk, dzm, dzw, zm, zw)
ratio = (terms["align_wrist"] / terms["align_main"]).item()
assert 1 / 3 < ratio < 3, f"표준화 공간 align 항 크기 불일치 (ratio {ratio:.3f})"
print(f"(B) N3 OK: raw std 비율 {r_raw:.2f}× → 표준화 후 {r_std:.3f}× "
      f"(align_w/align_m = {ratio:.3f})")

# ── (C) 왕복 + ζ 통화 일관성 ────────────────────────────────────────────────
inv_m = sm * ae_std.dz_std_main
inv_w = sw * ae_std.dz_std_wrist
assert torch.allclose(inv_m, dzm, atol=1e-6) and torch.allclose(inv_w, dzw, atol=1e-6), \
    "std_dz 역변환(σ 곱) 왕복 실패"
ae_std.eval()
with torch.no_grad():
    # decode 는 표준화 공간 소비: decode(std_dz(Δz)) == h([Δz/σ], z) 수동 계산
    a1 = ae_std.decode(torch.cat(ae_std.std_dz(dzm, dzw), 1), zm, zw)
    a2 = ae_std.h(torch.cat([dzm / ae_std.dz_std_main, dzw / ae_std.dz_std_wrist], 1),
                  torch.cat([zm, zw], 1))
    assert torch.equal(a1, a2), "decode(std_dz) ≠ h(z-score 수동) — 경계 불일치"
    # ζ 통화: encode(표준화 공간 출력) → decode(무스케일 직결) = phase2/rollout 경로
    zeta = ae_std.encode(chunk, zm, zw)
    _ = ae_std.decode(zeta, zm, zw)                       # 스케일 재적용 없이 통과
    # losses 의 recon-h 입력 == decode 경로 (train/eval 일관)
    _, parts_std = ae_std.losses(chunk, dzm, dzw, zm, zw, w)
    from torch.nn.functional import l1_loss
    assert abs(parts_std["recon"] - l1_loss(a1, chunk).item()) < 1e-6, \
        "losses recon ≠ decode(std_dz) 경로 (train_phase1 평가부 전사 불일치)"
print("(C) 왕복/통화 일관성 OK: std_dz↔σ곱 exact, decode=표준화 공간 소비, "
      "encode→decode 무스케일 직결")

# ── (D) N4: per-dim x0_std 가 블록 분산 보존 ────────────────────────────────
dc = DM + DW
torch.manual_seed(77)
fp_pd = FlowPolicy(d_model=128, layers=2, n_tokens=6, latent_dim=dc,
                   source="noise", x0_per_dim=True)
assert fp_pd.x0_std.shape == torch.Size([dc]), fp_pd.x0_std.shape
lt = torch.cat([dzm, dzw], 1)                # [ζ_main;ζ_wrist] 동형 (6.5× 블록 불균형)
fp_pd.x0_std.copy_(lt.std(0).clamp_min(1e-8))          # train_phase2 run_dual 전사
fp_sc = FlowPolicy(d_model=128, layers=2, n_tokens=6, latent_dim=dc, source="noise")
fp_sc.x0_std.fill_(lt.std().item())
gen = torch.Generator().manual_seed(11)
x0_pd = torch.randn((4096, dc), generator=gen) * fp_pd.x0_std
x0_sc = torch.randn((4096, dc), generator=gen) * fp_sc.x0_std
tgt_ratio = (lt[:, :DM].std() / lt[:, DM:].std()).item()
pd_ratio = (x0_pd[:, :DM].std() / x0_pd[:, DM:].std()).item()
sc_ratio = (x0_sc[:, :DM].std() / x0_sc[:, DM:].std()).item()
assert abs(pd_ratio - tgt_ratio) / tgt_ratio < 0.1, \
    f"N4 x0 블록비 {pd_ratio:.2f} ≠ 타깃 {tgt_ratio:.2f}"
assert abs(sc_ratio - 1.0) < 0.1, f"스칼라판 대조 실패 ({sc_ratio:.2f})"
with torch.no_grad():
    fp_pd.eval()
    out = fp_pd(toks, generator=torch.Generator().manual_seed(3))
    assert out.shape == (B, dc), out.shape                # per-dim broadcast 순전파 정상
print(f"(D) N4 OK: 타깃 블록 std 비 {tgt_ratio:.2f}× → per-dim x0 {pd_ratio:.2f}× 보존 "
      f"(단일 스칼라 {sc_ratio:.2f}× = 오염 재현)")

# ── (E) W-C 토큰열 소스 동형성: train(run_dual) vs rollout_sim vs rollout_dataset ──
tp2 = (WS / "src/training/train_phase2.py").read_text()
rs = (WS / "src/eval_libero/rollout_sim.py").read_text()
rd = (WS / "src/eval_libero/rollout_dataset.py").read_text()
assert "toks = [_pad_to(zp_m, dc), _pad_to(zc_m, dc), a_emb, _pad_to(ws, dc)]" in tp2, \
    "train run_dual W-C 토큰열 변형"
assert "toks = [_pad(zp), _pad(zc), a_emb, _pad(zw_sig)]" in rs, \
    "rollout_sim W-C 토큰열 변형"
assert "toks = [_pad(z_prev), _pad(z_cur), a_emb, _pad(zw_sig)]" in rd, \
    "rollout_dataset W-C 토큰열 변형"
# 세 경로 모두 [zp, zc, a_emb(#2=A_EMB_IDX), zw_sig] (+lang 마지막) — 동일 순서 확인
print("(E) W-C 토큰열 소스 동형 OK: train/rollout_sim/rollout_dataset = "
      "[zp, zc, a_emb, zw_sig](+lang)")

print("\nW-C SMOKE OK — (A) 기본 경로 HEAD 비트 동형 / (B) N3 스케일 정렬 / "
      "(C) 왕복·통화 일관 / (D) N4 블록 분산 보존 / (E) W-C 토큰열 3경로 동형")
