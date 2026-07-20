# libero.py build_policy_samples byte-identity 검증 (실 hdf5 2 에피소드, fake 앵커 — 모델 불요).
#
#   (1) 플래그 부재 (기본/obs/dual 경로): HEAD libero.py vs 작업트리 — 출력 배열 전부
#       np.array_equal (byte-identical 불변식).
#   (2) W-B obs_delta=True: 선행 배열은 flag-off 와 동일 + 추가 1배열 =
#       D[t].mean(0) − D[max(t−span,0)].mean(0) 수동 공식과 exact 일치.
#   (3) W-C wrist_cond_anchor: dual 8배열 동일 + 9번째 = main 앵커 wrist cur 수동 stack.
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "src"))
SCRATCH = Path(__file__).resolve().parent


def _load_head(rel, name):
    src = subprocess.run(["git", "-C", str(WS), "show", f"HEAD:{rel}"],
                         capture_output=True, text=True, check=True).stdout
    p = SCRATCH / f"_head_{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(f"head_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


head_lib = _load_head("src/data/libero.py", "libero")
from data import libero as new_lib  # noqa: E402


class FakePooled:
    """결정론 pooled 앵커 (이미지 통계 기반) — 인코더/GPU 불요."""
    def __init__(self, key, dim=8):
        self.cache_key, self.dim = key, dim

    def encode_images(self, imgs):
        out = []
        for im in imgs:
            a = np.asarray(im, dtype=np.float64)
            v = [a.mean(), a.std(), a[..., 0].mean(), a[..., 1].mean(),
                 a[..., 2].mean(), a[:64].mean(), a[64:].mean(), a[:, :64].mean()]
            out.append(np.array(v[:self.dim], dtype=np.float32))
        return {"embeds": np.stack(out)}


class FakeDense(FakePooled):
    """결정론 dense 앵커 (P=4 patch) — pool2 캐시 동형."""
    def encode_images(self, imgs):
        out = []
        for im in imgs:
            a = np.asarray(im, dtype=np.float64)
            qs = [a[:64, :64], a[:64, 64:], a[64:, :64], a[64:, 64:]]
            out.append(np.stack([
                np.array([q.mean(), q.std(), q[..., 0].mean(), q[..., 1].mean(),
                          q[..., 2].mean(), q.min(), q.max(), q[..., 2].std()],
                         dtype=np.float32)[:self.dim] for q in qs]))
        return {"tokens": np.stack(out)}


tmp = tempfile.mkdtemp(prefix="libero_byteid_")
cfg = {"data": {"source": "libero",
                "root": str(Path("~/clip_ws/data/libero/libero_spatial").expanduser()),
                "camera": "agentview_rgb", "wrist_camera": "eye_in_hand_rgb",
                "chunk_sec": 0.8, "n_chunk": 16, "val_episodes": 0.2,
                "cache_dir": tmp}}
ds_h, ds_n = head_lib.LiberoDataset(cfg), new_lib.LiberoDataset(cfg)
files = ds_h.episode_files()[:2]
main_a, wrist_a, dense_a = FakePooled("fake-main"), FakePooled("fake-wrist"), \
    FakeDense("fake-pool2")


def _eq(a, b):
    return len(a) == len(b) and all(
        len(x) == len(y) and all(np.array_equal(u, v) for u, v in zip(x, y))
        for x, y in zip(a, b))


# (1a) 기본 경로 (wrist 토큰 포함)
o_h = ds_h.build_policy_samples(main_a, files, stride=2)
o_n = ds_n.build_policy_samples(main_a, files, stride=2)
assert _eq(o_h, o_n), "기본 경로 출력 상이 (byte-identity 위반)"
# (1b) obs(grid dense) 경로 flag-off
obs = [("fake", dense_a, "eye_in_hand_rgb")]
g_h = ds_h.build_policy_samples(main_a, files, stride=2, obs_anchors=obs)
g_n = ds_n.build_policy_samples(main_a, files, stride=2, obs_anchors=obs)
assert _eq(g_h, g_n), "obs dense 경로 출력 상이"
# (1c) dual 경로 flag-off
d_h = ds_h.build_policy_samples(main_a, files, stride=2, wrist_anchor=wrist_a)
d_n = ds_n.build_policy_samples(main_a, files, stride=2, wrist_anchor=wrist_a)
assert _eq(d_h, d_n), "dual 경로 출력 상이"
print("(1) build_policy_samples flag-off == HEAD: 기본/obs/dual 전 경로 byte-identical OK")

# (2) W-B obs_delta: 선행 배열 불변 + 마지막 배열 = 수동 공식
w_n = ds_n.build_policy_samples(main_a, files, stride=2, obs_anchors=obs,
                                obs_delta=True)
span = ds_n.span
for ei, ep in enumerate(files):
    assert len(w_n[ei]) == len(g_n[ei]) + 1, "obs_delta 배열 수 불일치"
    assert all(np.array_equal(w_n[ei][k], g_n[ei][k]) for k in range(len(g_n[ei]))), \
        "obs_delta=True 가 선행 배열을 변형"
    D = ds_n.dense_embeddings(dense_a, ep, "eye_in_hand_rgb")
    T = len(ds_n.load_actions(ep))
    starts = list(range(0, T - span, 2))
    manual = np.stack([D[t].mean(0) - D[max(t - span, 0)].mean(0)
                       for t in starts]).astype(np.float32)
    assert np.array_equal(w_n[ei][-1], manual), "Δz̄_w 공식 불일치"
    assert np.all(w_n[ei][-1][0] == 0), "t=0 클램프(Δ=0) 위반"
print(f"(2) W-B obs_delta OK: 선행 배열 불변, Δz̄_w = p̄(t)−p̄(max(t−span,0)) exact "
      f"(span={span}, {len(files)} eps)")

# (3) W-C wrist_cond_anchor: dual 8배열 불변 + 9번째 = main 앵커 wrist cur
c_n = ds_n.build_policy_samples(main_a, files, stride=2, wrist_anchor=wrist_a,
                                wrist_cond_anchor=main_a)
for ei, ep in enumerate(files):
    assert len(c_n[ei]) == 9 and len(d_n[ei]) == 8
    assert all(np.array_equal(c_n[ei][k], d_n[ei][k]) for k in range(8)), \
        "wrist_cond_anchor 가 dual 배열을 변형"
    Zs = ds_n.embeddings(main_a, ep, "eye_in_hand_rgb")
    T = len(ds_n.load_actions(ep))
    starts = list(range(0, T - span, 2))
    assert np.array_equal(c_n[ei][8],
                          np.stack([Zs[t] for t in starts]).astype(np.float32)), \
        "zw_sig(9번째) 불일치"
print("(3) W-C wrist_cond_anchor OK: dual 8배열 불변, 9번째 = SigLIP2-wrist cur 동형")

print("\nLIBERO BYTE-ID OK — 실데이터 2 eps 에서 데이터측 불변식/신규 배열 전부 검증")
