"""LIBERO 데모 로더 — ActSimDataset과 동일 인터페이스 (임베딩 규격으로 통일).

데이터 형식 (robomimic HDF5, 태스크당 1파일 × 데모 50개):
  <suite>/<task>_demo.hdf5
    data/demo_K/obs/agentview_rgb : (T, H, W, 3) uint8, 20Hz
    data/demo_K/actions           : (T, 7)  — OSC 델타 (Δpos 3 + Δrot 3 + 그리퍼 1)

에피소드 단위 = (hdf5 경로, demo 키) 튜플. 언어 지시문은 태스크 파일명 기준으로
CLIP 텍스트 임베딩을 캐시한다 (저장만 — 정책 사용은 이후 단계).
"""
import os
import re
import zlib
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

HZ = 20.0


def _wrist_augment(arr, rng):
    """Exp2 학습용 약~중 photometric/geometric 증강 (동결 인코딩 전 이미지에 적용).

    복구 스펙(P2 +aug arm): 밝기/대비/색 ±0.2, p=0.5 GaussianBlur r0.5-1.5,
    p=0.5 회전 ±5°+0.9 center-crop-zoom, p=0.5 gaussian noise σ8.
    손목캠·3인칭 공용. 동료 복구본(siglip_ref_libero.py)과 동일.
    """
    img = Image.fromarray(arr)
    img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-0.2, 0.2))
    img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-0.2, 0.2))
    img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-0.2, 0.2))
    if rng.rand() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    if rng.rand() < 0.5:
        deg = rng.uniform(-5, 5); w, h = img.size
        img = img.rotate(deg, resample=Image.BILINEAR)
        cw, ch = int(w * 0.9), int(h * 0.9); l, t = (w - cw) // 2, (h - ch) // 2
        img = img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32)
    if rng.rand() < 0.5:
        a = a + rng.normal(0, 8.0, a.shape)          # σ≈0.03×255 (학습은 약하게)
    return np.clip(a, 0, 255).astype(np.uint8)


class LiberoDataset:
    def __init__(self, cfg):
        d = cfg["data"]
        roots = d["root"] if isinstance(d["root"], list) else [d["root"]]
        self.roots = [Path(os.path.expanduser(r)) for r in roots]
        self.camera = d.get("camera", "agentview_rgb")
        self.wrist_camera = d.get("wrist_camera")    # 예: eye_in_hand_rgb (없으면 미사용)
        self.chunk_sec = float(d["chunk_sec"])
        self.n_chunk = int(d["n_chunk"])
        self.cache_dir = Path(os.path.expanduser(d["cache_dir"]))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # dense(patch) 캐시 경로 — pooled와 분리 (F3/F4 dense 경로용, 기본 미사용).
        self.dense_cache_dir = Path(os.path.expanduser(
            d.get("dense_cache_dir", str(self.cache_dir / "dense"))))
        self.span = max(2, int(round(self.chunk_sec * HZ)))
        self.stride = max(1, self.span // 8)
        # Exp2 both-aug (P2). data.augment={view:N, wrist:N}. 없거나 0이면 완전
        # 무증강 → build_policy_samples 출력이 기존과 byte-identical (핵심 불변식).
        # N>0이면 해당 카메라 Z가 (T,N,D) variant 뱅크로(variant0=클린) 확장된다.
        aug = d.get("augment") or {}
        self.aug_view = int(aug.get("view", 0) or 0)
        self.aug_wrist = int(aug.get("wrist", 0) or 0)

    # ---------- 에피소드 열거: (파일, demo키) ----------

    def episode_files(self):
        eps = []
        for r in self.roots:
            for f in sorted(r.glob("*.hdf5")):
                with h5py.File(f, "r") as h:
                    demos = sorted(h["data"].keys(),
                                   key=lambda k: int(k.split("_")[-1]))
                eps += [(f, k) for k in demos]
        return eps

    @staticmethod
    def _key(ep):
        path, demo = ep
        return f"{path.stem}_{demo}"

    # ---------- 원시 접근 ----------

    def load_actions(self, ep):
        path, demo = ep
        with h5py.File(path, "r") as h:
            return h[f"data/{demo}/actions"][:].astype(np.float64)

    def load_frames(self, ep, camera=None):
        path, demo = ep
        with h5py.File(path, "r") as h:
            return h[f"data/{demo}/obs/{camera or self.camera}"][:]

    def instruction(self, ep):
        """태스크 파일명 → 자연어 지시문 (예: pick_up_the_..._demo.hdf5)."""
        path, _ = ep
        name = re.sub(r"_demo$", "", path.stem)
        # SCENE 접두어 제거 (예: LIVING_ROOM_SCENE1_)
        name = re.sub(r"^[A-Z0-9_]+SCENE\d+_", "", name)
        return name.replace("_", " ")

    # ---------- CLIP 임베딩 캐시 ----------

    def _emb_cache(self, ep, camera, clip):
        """앵커별 캐시 경로. 기본 CLIP(joint/norm)·cache_key 없는 인코더는 기존
        평면 캐시(하위호환), 그 외 앵커는 cache_key 하위 디렉터리로 분리."""
        legacy = self.cache_dir / (self._key(ep) + f"_{camera}.npz")
        key = getattr(clip, "cache_key", None)
        if key is None or key == "clip-vit-l-14/joint/norm":
            return legacy
        d = self.cache_dir / key
        d.mkdir(parents=True, exist_ok=True)
        return d / (self._key(ep) + f"_{camera}.npz")

    def embeddings(self, clip, ep, camera=None):
        camera = camera or self.camera
        cache = self._emb_cache(ep, camera, clip)
        if cache.exists():
            Z = np.load(cache)["Z"]
            _dim = getattr(clip, "dim", None)   # cowork §3: 캐시키 충돌(model-agnostic id) 즉시 검출
            assert _dim is None or Z.shape[-1] == _dim, \
                f"pooled cache dim {Z.shape[-1]} != anchor.dim {_dim} ({cache}) — cache-key 충돌?"
            return Z
        frames = [Image.fromarray(im) for im in self.load_frames(ep, camera)]
        Z = []
        for i in range(0, len(frames), 64):
            Z.append(clip.encode_images(frames[i:i + 64])["embeds"])
        Z = np.concatenate(Z)
        np.savez_compressed(cache, Z=Z)
        return Z

    def dense_embeddings(self, clip, ep, camera=None):
        """patch(dense) 토큰 캐시 [T, n_patch, d] — pooled(embeddings)와 분리 키.

        F3/F4 dense 경로용. 기본 phase1/phase2 파이프라인은 호출하지 않으므로
        pooled 경로 수치에 영향 없음 (대용량이라 옵션·서브셋부터).
        """
        camera = camera or self.camera
        # 앵커 cache_key 하위 디렉터리로 분리 (DINOv2-reg vs siglip2 dense 충돌 방지).
        key = getattr(clip, "cache_key", None)
        d = (self.dense_cache_dir / key) if key else self.dense_cache_dir
        cache = d / (self._key(ep) + f"_{camera}.npz")
        if cache.exists():
            D = None
            try:
                D = np.load(cache)["D"]
            except Exception:
                cache.unlink(missing_ok=True)   # 손상/부분 기록 캐시 → 아래서 재생성 (self-healing)
            if D is not None:
                _dim = getattr(clip, "dim", None)   # cowork §3: 차원불일치(캐시키 충돌)는 loud fail (재생성으로 숨기지 않음)
                assert _dim is None or D.shape[-1] == _dim, \
                    f"dense cache dim {D.shape[-1]} != anchor.dim {_dim} ({cache}) — cache-key 충돌?"
                return D
        d.mkdir(parents=True, exist_ok=True)
        frames = [Image.fromarray(im) for im in self.load_frames(ep, camera)]
        D = []
        for i in range(0, len(frames), 64):
            D.append(clip.encode_images(frames[i:i + 64])["tokens"])
        D = np.concatenate(D)
        tmp = cache.with_name(cache.name + ".tmp")     # 원자적 기록: crash 시 부분파일이 최종 경로에 남지 않도록
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, D=D)
        tmp.replace(cache)
        return D

    # ---------- Exp2 both-aug: M-variant 증강 임베딩 뱅크 ----------

    def _aug_cache(self, ep, camera, clip, variants):
        """증강 variant 뱅크 캐시 경로. 클린 캐시(_emb_cache)와 파일명 분리
        (_aug{M}), 앵커 cache_key 하위 디렉터리 규약은 _emb_cache와 동일 →
        클린 캐시 절대 미변경(가드레일)."""
        fname = self._key(ep) + f"_{camera}_aug{variants}.npz"
        key = getattr(clip, "cache_key", None)
        if key is None or key == "clip-vit-l-14/joint/norm":
            return self.cache_dir / fname
        d = self.cache_dir / key
        d.mkdir(parents=True, exist_ok=True)
        return d / fname

    def cam_aug_embeddings(self, clip, ep, camera, variants=3):
        """카메라 M-variant 증강 임베딩 뱅크 <key>_<cam>_aug{M}.npz — (T, M, D).

        variant0 = 클린(=embeddings() 결과, 오늘 캐시 그대로 재사용 → variant0가
        기존 임베딩과 정확히 동일). 1..M-1 = 동결 인코더 통과 전 이미지 증강본.
        에피소드×카메라별 결정적 RNG(crc32(key+camera)) → 재현·카메라별 독립.
        복구본(siglip_ref_libero.py) 스킴 미러: m=0은 rng 미소비, m>=1이 순서대로
        rng 소비. 클린 캐시 미변경.
        """
        cache = self._aug_cache(ep, camera, clip, variants)
        if cache.exists():
            try:
                return np.load(cache)["Z"]
            except Exception:
                cache.unlink(missing_ok=True)   # 손상/부분 기록(동시 arm race) → 아래서 재생성 (self-healing)
        Z0 = self.embeddings(clip, ep, camera)           # (T, D) 클린 == variant0
        frames = self.load_frames(ep, camera)            # (T,H,W,3) uint8
        rng = np.random.RandomState(
            zlib.crc32((self._key(ep) + camera).encode()) & 0xffffffff)
        vs = [Z0]
        for _m in range(1, int(variants)):
            imgs = [Image.fromarray(_wrist_augment(f, rng)) for f in frames]
            Z = []
            for i in range(0, len(imgs), 64):
                Z.append(clip.encode_images(imgs[i:i + 64])["embeds"])
            vs.append(np.concatenate(Z))
        Z = np.stack(vs, axis=1)                         # (T, M, D)
        tmp = cache.with_name(cache.name + ".tmp")       # 원자적 기록(동시 arm race 시 부분파일 방지)
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, Z=Z)
        tmp.replace(cache)
        return Z

    def instruction_embedding(self, clip, ep):
        path, _ = ep
        # 앵커별 텍스트 공간 분리(CLIP 768d vs SigLIP2 1152d). CLIP joint/norm은
        # 평면 캐시 유지(하위호환·bit-identity), 그 외 앵커는 cache_key 하위로.
        key = getattr(clip, "cache_key", None)
        if key is None or key == "clip-vit-l-14/joint/norm":
            cache = self.cache_dir / (path.stem + "_lang.npz")
        else:
            d = self.cache_dir / key
            d.mkdir(parents=True, exist_ok=True)
            cache = d / (path.stem + "_lang.npz")
        if cache.exists():
            return np.load(cache)["L"]
        L = clip.encode_texts([self.instruction(ep)])["embeds"][0]
        np.savez_compressed(cache, L=L)
        return L

    # ---------- 학습 쌍 생성 (act_sim과 동일 수식) ----------

    def resample_chunk(self, seg):
        src = np.linspace(0, len(seg) - 1, self.n_chunk)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, len(seg) - 1)
        w = (src - lo)[:, None]
        return seg[lo] * (1 - w) + seg[hi] * w

    def build(self, clip, files=None, verbose=True):
        files = files or self.episode_files()
        out = []
        for ep in files:
            acts = self.load_actions(ep)
            T = len(acts)
            Z = self.embeddings(clip, ep)
            starts = list(range(0, T - self.span, self.stride))
            Zt = np.stack([Z[t] for t in starts])
            Ztn = np.stack([Z[t + self.span] for t in starts])
            A = np.stack([self.resample_chunk(acts[t:t + self.span]).ravel()
                          for t in starts])
            out.append((Zt.astype(np.float32), Ztn.astype(np.float32),
                        A.astype(np.float32)))
            if verbose:
                print(f"  {self._key(ep)}: T={T}, pairs {len(starts)}")
        return out

    def build_policy_samples(self, clip, files=None, stride=2, obs_anchors=None,
                             f4_anchor=None):
        """연속 윈도우 삼중쌍 (경계 포함 — 롤아웃 부트스트랩 분포 커버).

        obs_anchors: [(name, anchor, camera), ...] (F3 obs-fusion). 주어지면 각
        관측 앵커의 dense patch 토큰 D_cur[t] (Zc와 동일 starts 정렬)을 손목캠
        배열 뒤에 순서대로 덧붙인다. None(기본)이면 출력은 기존과 완전 동일.

        f4_anchor: (anchor, camera) (C1/F4 fine 채널). 주어지면 patch ΔF =
        D[t+span] − D[t] (동일 인덱스 patch 차분, agentview 정적 카메라 전제 — D0)를
        (n, P, dense_dim)로 맨 뒤에 덧붙인다. None(기본)이면 출력 불변.

        Exp2 both-aug (data.augment): aug_view=N/aug_wrist=N>0이면 해당 카메라
        Z를 (n, M, D) variant 뱅크로 반환(variant0=클린). 둘 다 0(기본)이면 아래
        else 경로만 타 출력이 기존과 완전 동일(byte-identical). 학습측 variant
        선택은 train_phase2가 담당(train=샘플별 랜덤, val/eval=variant0).
        """
        view_aug, wrist_aug = self.aug_view, self.aug_wrist
        files = files or self.episode_files()
        out = []
        for ep in files:
            acts = self.load_actions(ep)
            T = len(acts)
            starts = list(range(0, T - self.span, stride))

            def past_seg(t):
                if t == 0:
                    return np.repeat(acts[0:1], 2, axis=0)
                return acts[max(t - self.span, 0):t]

            if view_aug:                             # 3인칭 증강 뱅크 (T,M,D)
                Za = self.cam_aug_embeddings(clip, ep, self.camera,
                                             variants=view_aug)
                Zp = np.stack([Za[max(t - self.span, 0)] for t in starts])  # (n,M,D)
                Zc = np.stack([Za[t] for t in starts])
                Zn = np.stack([Za[t + self.span] for t in starts])
            else:
                Z = self.embeddings(clip, ep)
                Zp = np.stack([Z[max(t - self.span, 0)] for t in starts])
                Zc = np.stack([Z[t] for t in starts])
                Zn = np.stack([Z[t + self.span] for t in starts])
            Ap = np.stack([self.resample_chunk(past_seg(t)).ravel()
                           for t in starts])
            Af = np.stack([self.resample_chunk(acts[t:t + self.span]).ravel()
                           for t in starts])
            arrs = [Zp, Zc, Zn, Ap, Af]
            if self.wrist_camera:                    # 6번째: 손목캠 z_t (정책 토큰용)
                if wrist_aug:
                    Zw = self.cam_aug_embeddings(clip, ep, self.wrist_camera,
                                                 variants=wrist_aug)          # (T,M,D)
                    arrs.append(np.stack([Zw[t] for t in starts]))           # (n,M,D)
                else:
                    Zw = self.embeddings(clip, ep, self.wrist_camera)
                    arrs.append(np.stack([Zw[t] for t in starts]))
            for _name, anchor, cam in (obs_anchors or []):   # F3: 관측 앵커별 dense
                D = self.dense_embeddings(anchor, ep, cam)   # [T, P, d]
                arrs.append(np.stack([D[t] for t in starts]))
            if f4_anchor is not None:                        # C1/F4: patch ΔF (동일인덱스 차분)
                f4_anc, f4_cam = f4_anchor
                D = self.dense_embeddings(f4_anc, ep, f4_cam)   # [T, P, d]
                arrs.append(np.stack([D[t + self.span] - D[t] for t in starts]))
            out.append(tuple(x.astype(np.float32) for x in arrs))
        return out


if __name__ == "__main__":
    # 로더 단독 점검: python src/data/libero.py
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import yaml
    from core.clip_wrapper import ClipWrapper

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2]
                              / "configs" / "phase1_libero.yaml"))
    ds = LiberoDataset(cfg)
    eps = ds.episode_files()
    print(f"episodes: {len(eps)} (파일 {len(set(p for p, _ in eps))}개 태스크)")
    clip = ClipWrapper()
    print("지시문 예:", ds.instruction(eps[0]))
    pairs = ds.build(clip, eps[:2])
    Zt, Ztn, A = pairs[0]
    print(f"pair: z {Zt.shape}, chunk {A.shape} (span {ds.span} steps @ {HZ}Hz)")
