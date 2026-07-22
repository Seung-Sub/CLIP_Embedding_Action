# P-B LangSelPool + capacity-sweep 스모크 (CPU, 합성 텐서, ckpt/캐시/모델다운로드 불요).
#
# 검증 8단:
#   (A) byte-identity vs HEAD: git HEAD 의 obs_fusion.py/networks.py 를 로드해
#       (1) GridObs — 동일 시드 생성 시 state_dict 완전 동일 + eval forward 완전 동일
#           (LangSelPool 클래스 "추가"가 기존 모듈의 RNG/키에 무영향 증명)
#       (2) DeltaAE(hidden_g/hidden_h 미지정) — state_dict 키+값 완전 동일 + losses()/
#           g/h 출력 완전 동일 (capacity-sweep 노브 기본값 비트 동형).
#   (B) capacity 노브 배선: hidden_g=256 → g 만 좁아짐 / hidden_h=1024 → h 만 넓어짐.
#   (C) LangSelPool 단위: shape (B,K,out) · eval 결정론 · init std≈0.02 · n_patch
#       불일치 loud-fail · return_attn (B,heads,K,P).
#   (D) 언어 인과성/선택 실재성 배선: lang 교체 → 토큰 변화, patch 셔플 → 토큰 변화
#       (P3/P4 probe 가 측정할 채널이 실제로 배선돼 있는지의 전제 확인).
#   (E) VERIFY A1 loud-fail (기능 검증): train_phase2._validate_patch_obs_cfg 에 bogus
#       키/전제 위반 → AssertionError, 블록 부재 → None(no-op).
#   (F) 토큰-순서 (VERIFY A2 확장): train_phase2 전사 vs rollout 전사 스택 완전 동일 +
#       canonical 순서 base(5) → langsel(K, 마지막). B1 토큰 위치 = #6.
#   (G) 소스-순서/배선 검사: patch 분기 위치(grid 뒤·wdelta 앞·stack 앞), n_tokens Kp
#       포함, save_dict guarded, Siglip2Anchor patch_dim 수리 라인 존재.
#   (H) config 정합: B1/B2-STUB yaml 의 module 블록이 가드를 통과.
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
import yaml

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


head_obs = _load_head("src/models/obs_fusion.py", "pb_obs_fusion")
head_net = _load_head("src/models/networks.py", "pb_networks")
from models.networks import DeltaAE  # noqa: E402  (작업트리 신판)
from models.obs_fusion import GridObs, LangSelPool  # noqa: E402

B, PD, TD, OD, P, K = 2, 96, 64, 96, 16, 1

# ── (A1) GridObs: LangSelPool 추가가 기존 모듈에 무영향 (state_dict + forward 동형) ──
torch.manual_seed(11)
g_head = head_obs.GridObs(patch_dim=PD, out_dim=OD, n_tokens=4, pool="avg",
                          ln=True, tok_drop=0.1, group_drop=0.1, init_std=0.02)
torch.manual_seed(11)
g_new = GridObs(patch_dim=PD, out_dim=OD, n_tokens=4, pool="avg",
                ln=True, tok_drop=0.1, group_drop=0.1, init_std=0.02)
sd_h, sd_n = g_head.state_dict(), g_new.state_dict()
assert list(sd_h.keys()) == list(sd_n.keys()) and \
    all(torch.equal(sd_h[k], sd_n[k]) for k in sd_h), "GridObs state_dict 변형"
pt = torch.randn(B, 4, PD, generator=torch.Generator().manual_seed(5))
g_head.eval(); g_new.eval()
with torch.no_grad():
    assert torch.equal(g_head(pt), g_new(pt)), "GridObs forward 출력 상이"
print("(A1) GridObs == HEAD: state_dict/forward 비트 동형 OK (LangSelPool 추가 무영향)")

# ── (A2) DeltaAE 노브 미지정 = HEAD 와 비트 동형 (capacity sweep byte-identity) ──
ACT, NC, LAT = 7, 16, 128
torch.manual_seed(22)
ae_head = head_net.DeltaAE(ACT, NC, LAT, hidden=64, layers=2)
torch.manual_seed(22)
ae_new = DeltaAE(ACT, NC, LAT, hidden=64, layers=2)     # hidden_g/hidden_h 미지정
sd_h, sd_n = ae_head.state_dict(), ae_new.state_dict()
assert list(sd_h.keys()) == list(sd_n.keys()) and \
    all(torch.equal(sd_h[k], sd_n[k]) for k in sd_h), "DeltaAE state_dict 변형"
gen = torch.Generator().manual_seed(6)
chunk = torch.randn(B, NC, ACT, generator=gen)
dz = torch.randn(B, LAT, generator=gen)
zt = torch.randn(B, LAT, generator=gen)
w = {"align": 0.5, "recon": 0.5, "cycle": 0.25}
ae_head.eval(); ae_new.eval()
with torch.no_grad():
    th, ph = ae_head.losses(chunk, dz, w, zt)
    tn, pn = ae_new.losses(chunk, dz, w, zt)
    assert torch.equal(th, tn) and ph == pn, "DeltaAE losses 출력 상이"
    assert torch.equal(ae_head.g(chunk, zt), ae_new.g(chunk, zt))
    assert torch.equal(ae_head.h(dz, zt), ae_new.h(dz, zt))
print("(A2) DeltaAE knob-off == HEAD: state_dict/losses/g/h 비트 동형 OK")

# ── (B) capacity 노브: g/h 폭 독립 오버라이드 ──
ae_g = DeltaAE(ACT, NC, LAT, hidden=64, layers=2, hidden_g=32)
ae_h = DeltaAE(ACT, NC, LAT, hidden=64, layers=2, hidden_h=128)
assert ae_g.g.conv[0].out_channels == 32 and ae_g.h.mlp[1].out_features == 64, \
    "hidden_g 가 g 에만 적용되지 않음"
assert ae_h.g.conv[0].out_channels == 64 and ae_h.h.mlp[1].out_features == 128, \
    "hidden_h 가 h 에만 적용되지 않음"
with torch.no_grad():
    assert ae_g.h(dz, zt).shape == (B, NC, ACT) and \
        ae_h.g(chunk, zt).shape == (B, LAT)          # 폭 변경에도 입출력 규약 불변
print("(B) capacity 노브 OK: hidden_g→g 전용 / hidden_h→h 전용, 입출력 규약 불변")

# ── (C) LangSelPool 단위 검사 ──
torch.manual_seed(33)
lsp = LangSelPool(patch_dim=PD, text_dim=TD, out_dim=OD, n_patch=P, n_tokens=K,
                  d_attn=32, heads=4, tok_drop=0.1, group_drop=0.1)
with torch.no_grad():
    assert abs(lsp.kv_proj.weight.std().item() - 0.02) < 0.005 and \
        abs(lsp.pos_emb.std().item() - 0.02) < 0.005 and \
        abs(lsp.query_offset.std().item() - 0.02) < 0.01 and \
        float(lsp.out.bias.abs().max()) == 0.0, "init 규격(std 0.02/bias 0) 위반"
patches = torch.randn(B, P, PD, generator=torch.Generator().manual_seed(7))
lang = torch.randn(B, TD, generator=torch.Generator().manual_seed(8))
lsp.eval()
with torch.no_grad():
    t1 = lsp(patches, lang)
    t2 = lsp(patches, lang)
    tok, aw = lsp(patches, lang, return_attn=True)
assert t1.shape == (B, K, OD) and torch.equal(t1, t2), "shape/eval 결정론 위반"
# need_weights=True 는 MHA 가 math-path 로 전환(fast-path 커널과 부동소수 순서 상이)
# → 학습/롤아웃 경로(need_weights=False)와 수치 동일이 아니라 allclose 로 검증.
# 로깅 전용 경로이므로 무해 (토큰은 항상 need_weights=False 경로가 소비).
assert torch.allclose(tok, t1, atol=1e-5), "return_attn 경로 토큰 불일치(>1e-5)"
assert aw.shape == (B, 4, K, P), f"attn weights shape {tuple(aw.shape)} != (B,heads,K,P)"
assert torch.allclose(aw.sum(-1), torch.ones(B, 4, K), atol=1e-5), "attn 확률 비정규"
try:
    lsp(torch.randn(B, P + 1, PD), lang)
    raise SystemExit("FAIL: n_patch 불일치가 무음 통과")
except AssertionError:
    pass
print("(C) LangSelPool 단위 OK: init 0.02/bias0, (B,K,out), eval 결정론, "
      "attn (B,heads,K,P) 정규, n_patch loud-fail")

# ── (D) 언어 인과성/선택 배선 ──
lang2 = torch.randn(B, TD, generator=torch.Generator().manual_seed(9))
perm = torch.randperm(P, generator=torch.Generator().manual_seed(10))
with torch.no_grad():
    t_lang2 = lsp(patches, lang2)
    t_shuf = lsp(patches[:, perm], lang)
assert not torch.allclose(t1, t_lang2), "lang 교체에 토큰 불변 — 언어 경로 미배선"
assert not torch.allclose(t1, t_shuf), "patch 셔플에 토큰 불변 — kv/pos 경로 미배선"
# lang 스케일 불변성 (L2-norm 배선): lang×10 → 쿼리 동일 → 토큰 동일
with torch.no_grad():
    assert torch.allclose(t1, lsp(patches, lang * 10.0), atol=1e-5), \
        "lang L2-norm 미배선 (스케일 지배 방지 §4.4)"
print("(D) 언어 인과성 OK: lang→토큰 민감, patch→토큰 민감, lang 스케일 불변(L2-norm)")

# ── (E) VERIFY A1 loud-fail (기능 검증 — train_phase2 임포트) ──
sys.path.insert(0, str(WS / "src" / "training"))
import train_phase2 as tp2mod  # noqa: E402

ok_m = {"name": "flow", "lang_token": True,
        "patch_obs": {"anchor": {"name": "dinov3"}, "camera": "agentview_rgb",
                      "n_tokens": 1, "d_attn": 768, "heads": 8,
                      "tok_drop": 0.1, "group_drop": 0.1}}
assert tp2mod._validate_patch_obs_cfg(ok_m) == ok_m["patch_obs"], "정상 블록 통과 실패"
assert tp2mod._validate_patch_obs_cfg({"name": "flow"}) is None, "블록 부재가 None 아님"
for bad, msg in [
    ({**ok_m, "patch_obs": {**ok_m["patch_obs"], "tok_dorp": 0.1}}, "bogus 키"),
    ({**ok_m, "lang_token": False}, "lang_token 부재"),
    ({**ok_m, "name": "mlp"}, "mlp 정책"),
    ({**ok_m, "grid_obs": {"anchor": {}}}, "grid_obs 병용"),
    ({**ok_m, "obs": {"encoders": []}}, "obs 병용"),
    ({**ok_m, "f4": {"enable": True}}, "f4 병용"),
    ({**ok_m, "dual_stream": True}, "dual_stream"),
]:
    try:
        tp2mod._validate_patch_obs_cfg(bad)
        raise SystemExit(f"FAIL: {msg} 가 무음 통과 (VERIFY A1 위반)")
    except AssertionError:
        pass
print("(E) VERIFY A1 loud-fail OK: bogus 키/전제 위반 7종 전부 AssertionError, 부재=None")

# ── (F) 토큰-순서 (VERIFY A2 확장): train 전사 vs rollout 전사 + canonical 위치 ──
D = OD                                           # B1: latent = lang = wrist 폭 동일(1024 축소판)
torch.manual_seed(44)
mod = LangSelPool(patch_dim=PD, text_dim=D, out_dim=D, n_patch=P, n_tokens=K,
                  d_attn=32, heads=4).eval()
zp_c, zc_c = torch.full((1, D), 1.0), torch.full((1, D), 2.0)
aemb, lang_t, wr_c = (torch.full((1, D), 3.0), torch.full((1, D), 4.0),
                      torch.full((1, D), 5.0))
pat = torch.randn(1, P, PD, generator=torch.Generator().manual_seed(12))
_pad = lambda t: t                                # 전 토큰 동일 폭 → pad no-op  # noqa: E731
with torch.no_grad():
    # train측 전사 (train_phase2.forward: base → patch append → stack)
    base = [_pad(zp_c), _pad(zc_c), aemb, _pad(lang_t), _pad(wr_c)]
    toks = list(base)
    patch_tok = mod(pat, lang_t)
    toks = toks + [patch_tok[:, k] for k in range(K)]
    train_stack = torch.stack(toks, dim=1)
    # rollout측 전사 (rollout_sim: toks 리스트 → patch_toks append → pad → stack)
    toks_r = [zp_c, zc_c, aemb, lang_t, wr_c]
    pt_r = mod(pat, lang_t)
    toks_r = toks_r + [pt_r[:, k] for k in range(pt_r.size(1))]
    toks_r = [t if t.shape[-1] == D else
              torch.nn.functional.pad(t, (0, D - t.shape[-1])) for t in toks_r]
    rollout_stack = torch.stack(toks_r, dim=1)
assert train_stack.shape == rollout_stack.shape == (1, 5 + K, D)
assert torch.equal(train_stack, rollout_stack), "train/rollout 토큰 스택 불일치"
means = train_stack[0, :5].mean(dim=-1)
assert torch.allclose(means, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), means
assert torch.equal(train_stack[0, 5], patch_tok[0, 0]), \
    "B1 langsel 토큰이 canonical 위치(#6, base 뒤 마지막)가 아님"
print(f"(F) A2 확장 OK — train/rollout 스택 동형 (shape {tuple(train_stack.shape)}), "
      "canonical base(5)→langsel(#6 마지막)")

# ── (G) 소스-순서/배선 검사 (전사 드리프트 방호) ──
tp2 = (WS / "src/training/train_phase2.py").read_text()
rs = (WS / "src/eval_libero/rollout_sim.py").read_text()
rd = (WS / "src/eval_libero/rollout_dataset.py").read_text()
an = (WS / "src/core/anchor.py").read_text()
fwd = tp2[tp2.index("def forward(zp, zc, zn, aemb, cf, lang, wr, *dobs"):]
assert fwd.index("base = [_pad(zp_c)") \
    < fwd.index("[grid_tok[:, k] for k in range(Kg)]") \
    < fwd.index("patch_tok = patch_obs_mod(dobs[0], lang)") \
    < fwd.index("toks = toks + [_pad(wd)]") \
    < fwd.index("toks = torch.stack(toks, dim=1)"), \
    "train: base→grid→patch→w_tok→stack 순서 깨짐"
seg = rs[rs.index("toks = [zp_c, zc_c, a_emb]"):]
assert seg.index("toks = toks + grid_toks(obs)") \
    < seg.index("toks = toks + patch_toks(obs)") < seg.index("torch.stack(toks"), \
    "rollout_sim: grid→patch→stack 순서 깨짐"
segd = rd[rd.index("a_emb = ae.g(past_t, z_prev)"):]
assert segd.index("patch_tok = patch_obs(") < segd.index("if wdelta is not None"), \
    "rollout_dataset: patch 가 w_tok(마지막) 앞이 아님"
assert "n_tokens = 3 + int(use_lang) + int(use_wrist) + K + Kg + Kp + int(use_wdelta)" \
    in tp2, "train n_tokens 에 Kp 미포함"
assert "n_tokens=3 + int(use_lang) + int(use_wrist) + K + Kg + Kp" in rd, \
    "rollout n_tokens 에 Kp 미포함"
assert 'if patch_cfg:       # P-B LangSelPool best state — off면 키 자체 미생성(dict 불변)' \
    in tp2 and 'save_dict["patch_obs"]' in tp2, "ckpt patch_obs guarded 저장 부재"
assert "self.patch_dim = self.dim" in an.split("class Siglip2Anchor")[1] \
    .split("class Dinov2Anchor")[0], \
    "Siglip2Anchor patch_dim 수리(self.patch_dim = self.dim) 부재 — 롤아웃 1152 함정"
assert 'patch_cfg = _validate_patch_obs_cfg(m_cfg)' in tp2, "가드 호출 부재"
assert 'hidden_g=p1["model"].get("hidden_g")' in tp2 and \
    'hidden_g=p1["model"].get("hidden_g")' in rd, \
    "capacity 노브 phase2/rollout 재구성 배선 부재"
print("(G) 소스-순서/배선 OK: 삽입 위치·n_tokens·guarded 저장·patch_dim 수리·노브 배선")

# ── (H) config 정합: B1/B2 yaml 의 module 블록이 가드 통과 + 규격 값 ──
for cfg_name, sub in [("phase2_libero_large256_langselpool.yaml", "large256"),
                      ("phase2_libero_concat_langselpool_STUB.yaml", "concat")]:
    c = yaml.safe_load(open(WS / "configs" / cfg_name))
    pc = tp2mod._validate_patch_obs_cfg(c["module"])
    assert pc is not None and pc["n_tokens"] == 1 and pc["anchor"]["pool_to"] == 8 \
        and pc["camera"] == "agentview_rgb" and pc["tok_drop"] == 0.1 \
        and pc["group_drop"] == 0.1, f"{cfg_name}: patch_obs 규격 불일치"
    assert c["module"]["lang_token"] and c["module"]["name"] == "flow"
print("(H) config 정합 OK: B1/B2-STUB module 블록 가드 통과 "
      "(n_tokens=1, pool_to=8, main 카메라, drop 0.1/0.1)")

print("\nP-B SMOKE OK — (A) HEAD 비트 동형(GridObs/DeltaAE) / (B) capacity 노브 /"
      " (C,D) LangSelPool 단위·언어 인과성 / (E) VERIFY A1 loud-fail /"
      " (F) A2 토큰-순서 / (G) 소스 배선 / (H) config 정합")
