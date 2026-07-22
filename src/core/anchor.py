"""앵커(관측/텍스트 임베딩 백본) 추상화 — 계획서 Phase 0.1.

인터페이스 (모든 구현체 공통):
  encode_images(pil_images) -> {"embeds": (N, dim) float32, "tokens": (N, P, patch_dim)|None}
  encode_texts(texts)       -> {"embeds": (N, dim_text) float32}  (has_text=False면 예외)
  속성: dim / patch_dim / has_text / id
  옵션: projection {joint, pre} / normalize {true, false}
  cache_key: "{id}/{projection}/{norm|raw}" — 임베딩 캐시 디렉터리 분리 키

선택은 config의 anchor 섹션 (없으면 기본 = 기존과 동일):
  anchor: {name: clip|siglip2|dinov2, projection: joint, normalize: true, model_dir: ...}

기본값(ClipAnchor, joint, normalize=True)은 기존 ClipWrapper와 출력이 완전 동일
→ 기존 평면 캐시(libero_emb/*.npz 등) 재사용 가능 (로더의 하위호환 폴백 참조).
ImageBindAnchor는 비상업 라이선스로 구현 보류 (계획서 리스크 (iv) — SigLIP2로 대체).
"""
import numpy as np
import torch

from core.config import load_config


def _crop224(pil_images):
    """P-A center-crop 전처리 (DESIGN_pipeline_rethink §4.2(1)·§8 P-A).

    콜리그-EXACT 기하 규약 = facebook/dinov2-large 기본 AutoImageProcessor
    (BitImageProcessor — _Dinov2FusionBranch 1차 arm / Dinov2Anchor center_crop=True
    판이 쓰는 바로 그 경로)의 기하 연산부 verbatim 미러:
      resize(shortest_edge=256, resample=BICUBIC, int-내림 aspect 유지)
      → center_crop(224×224, offset=(h−224)//2 / (w−224)//2).
    이후 각 앵커의 native processor 가 model 해상도로 재-resize (crop → native resize).
    LIBERO 256×256 렌더: resize no-op → 테두리 12.5% 삭제(작업공간 중앙 확대)
    = 격리 실증 +~5pp 레버 (DINOv2-avg@crop 96.0 vs matched no-crop 90.5, §8).
    """
    from PIL import Image
    out = []
    for im in pil_images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if min(w, h) != 256:                       # HF get_size_with_aspect_ratio 미러
            short, long_ = (w, h) if w <= h else (h, w)
            new_long = int(256 * long_ / short)    # int-내림 (HF 동일)
            w, h = (256, new_long) if w <= h else (new_long, 256)
            im = im.resize((w, h), Image.BICUBIC)
        left, top = (w - 224) // 2, (h - 224) // 2   # HF center_crop 오프셋 규약
        out.append(im.crop((left, top, left + 224, top + 224)))
    return out


class BaseAnchor:
    id = "base"
    dim = None
    patch_dim = None
    has_text = False

    def __init__(self, projection="joint", normalize=True):
        assert projection in ("joint", "pre"), projection
        self.projection = projection
        self.normalize = normalize

    @property
    def cache_key(self):
        return f"{self.id}/{self.projection}/{'norm' if self.normalize else 'raw'}"

    def _post(self, x):
        x = x.float()
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x.cpu().numpy()

    def encode_texts(self, texts):
        raise RuntimeError(f"{self.id}: 텍스트 인코더 없음 (has_text=False) — "
                           "lang_token 조건은 언어 정렬 앵커에서만 가능")


class ClipAnchor(BaseAnchor):
    """frozen CLIP ViT-L/14. joint: projection 후 768 / pre: vision pooler 1024 (텍스트 768)."""
    id = "clip-vit-l-14"
    has_text = True
    patch_dim = 1024

    def __init__(self, projection="joint", normalize=True, cfg=None):
        super().__init__(projection, normalize)
        from core.clip_wrapper import ClipWrapper
        self._w = ClipWrapper(cfg if (cfg and "clip" in cfg) else None)
        self.dim = 768 if projection == "joint" else 1024
        self.dim_text = 768
        self.device = self._w.device

    @torch.no_grad()
    def encode_images(self, pil_images):
        if self.projection == "joint" and self.normalize:
            return self._w.encode_images(pil_images)      # 기존 경로와 완전 동일
        m, proc = self._w.model, self._w.processor
        inputs = proc(images=pil_images, return_tensors="pt").to(self.device)
        vout = m.vision_model(pixel_values=inputs["pixel_values"].to(m.dtype))
        pooled = (vout.pooler_output if self.projection == "pre"
                  else m.visual_projection(vout.pooler_output))
        tokens = (vout.last_hidden_state.float().cpu().numpy()
                  if self._w.save_tokens else None)
        return {"embeds": self._post(pooled), "tokens": tokens}

    @torch.no_grad()
    def encode_texts(self, texts):
        if self.projection == "joint" and self.normalize:
            return self._w.encode_texts(texts)
        m, proc = self._w.model, self._w.processor
        inputs = proc(text=texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(self.device)
        tout = m.text_model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])
        pooled = (tout.pooler_output if self.projection == "pre"
                  else m.text_projection(tout.pooler_output))
        return {"embeds": self._post(pooled), "tokens": None}


class Siglip2Anchor(BaseAnchor):
    """SigLIP2 so400m — 언어 정렬 앵커 후보 A군. joint(공유 공간)만 지원."""
    id = "siglip2-so400m"
    has_text = True
    patch_dim = 1152

    def __init__(self, projection="joint", normalize=True, model_dir=None, crop=False):
        if projection != "joint":
            raise ValueError("siglip2: projection=pre 미지원 (공유 공간 헤드 일체형)")
        super().__init__(projection, normalize)
        # P-A crop arm: encode 직전 콜리그-EXACT 224 center-crop (텍스트 경로 무관).
        # 인스턴스 id 접미사가 클래스 id 를 가림 → 캐시 키 완전 분리 (no-crop 캐시 불오염).
        self.crop = bool(crop)
        if self.crop:
            self.id = self.id + "-crop224"
        from transformers import AutoModel, AutoProcessor
        src = model_dir or "google/siglip2-so400m-patch14-384"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            src, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(src)
        self.dim = self.model.config.vision_config.hidden_size    # so400m=1152 / large256=1024
        # 롤아웃 patch_dim 함정 수리 (DESIGN_WD_WAprime_v1 §3.2 사전등록): 클래스 상수
        # patch_dim=1152 는 so400m 전용 — large256 실폭은 1024 라 rollout_dataset 이
        # ganc.patch_dim 으로 모듈을 재구성할 때 ckpt 와 shape 불일치(학습만 통과하고
        # 롤아웃에서 죽는 함정). 인스턴스 실폭으로 덮어씀 (so400m 은 1152=1152 no-op).
        self.patch_dim = self.dim
        self.dim_text = self.dim
        self.save_tokens = False                  # E3에서 True로 (패치 토큰 반환)

    @staticmethod
    def _tensor(out):
        """transformers 버전에 따라 get_*_features가 텐서 또는 출력 객체 반환."""
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def encode_images(self, pil_images):
        if self.crop:                              # P-A: crop → native resize 순서
            pil_images = _crop224(list(pil_images))
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        px = inputs["pixel_values"].to(self.model.dtype)
        emb = self._tensor(self.model.get_image_features(pixel_values=px))
        tokens = None
        if getattr(self, "save_tokens", False):    # E3: vision tower 패치 토큰 노출
            vout = self.model.vision_model(pixel_values=px)
            tokens = vout.last_hidden_state.float().cpu().numpy()   # (N, P, 1152)
        return {"embeds": self._post(emb), "tokens": tokens}

    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding="max_length",
                                truncation=True).to(self.device)
        emb = self._tensor(self.model.get_text_features(input_ids=inputs["input_ids"]))
        return {"embeds": self._post(emb), "tokens": None}


class Dinov2Anchor(BaseAnchor):
    """DINOv2-L — 무언어 대조 앵커 (H2).

    v2 보정 (앵커 적응 감사, 2026-07-08 — 검증 에이전트 발견):
    - center-crop 제거: 기본 processor는 256 resize→224 crop으로 시뮬 렌더의 테두리
      12.5%를 삭제 (DINO-WM 등 로봇 관행 = 224 직접 resize). → do_center_crop=False.
    - pooled 옵션: cls(기존) / clsmp(CLS ⊕ patch-mean concat, 2048d — DINOv2 논문
      프로빙 프로토콜의 강한 구성; DINO-WM 절제에서 CLS 단독은 dynamics에 유의 저하).
    - 캐시 키에 전처리 판 반영 (id 접미사) — 구 캐시(crop판)와 혼합 방지.
    """
    has_text = False
    patch_dim = 1024

    def __init__(self, projection="pre", normalize=True, model_dir=None,
                 pooled="cls", center_crop=False):
        super().__init__("pre", normalize)     # joint 공간 없음 → pre 고정
        from transformers import AutoImageProcessor, AutoModel
        src = model_dir or "facebook/dinov2-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            src, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        # with-registers 판 감지: last_hidden = [CLS, R개 레지스터, 패치…] (R=4).
        # 무레지스터(기본 dinov2-large)는 0 → 패치 slice가 [:, 1:]로 기존과 동일.
        self.num_registers = getattr(self.model.config, "num_register_tokens", 0) or 0
        if center_crop:
            self.processor = AutoImageProcessor.from_pretrained(src)
            self.id = "dinov2-large"                     # 구판 (crop) — 기존 캐시 호환
        else:
            self.processor = AutoImageProcessor.from_pretrained(
                src, do_center_crop=False,
                size={"height": 224, "width": 224}, do_resize=True)
            self.id = "dinov2-large-nc"                  # no-crop 판 = 새 캐시 키
        if self.num_registers:
            self.id += "-reg"                            # 레지스터 판 = 별도 dense 캐시 키
        assert pooled in ("cls", "clsmp"), pooled
        self.pooled = pooled
        if pooled == "clsmp":
            self.id += "-clsmp"
        self.dim = self.model.config.hidden_size * (2 if pooled == "clsmp" else 1)

    @torch.no_grad()
    def encode_images(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        out = self.model(pixel_values=inputs["pixel_values"].to(self.model.dtype))
        cls = out.pooler_output                          # = last_hidden[:, 0] (HF 검증)
        sl = 1 + self.num_registers                      # CLS(+레지스터) 제거 → 패치만
        if self.pooled == "clsmp":
            pm = out.last_hidden_state[:, sl:].mean(dim=1)
            cls = torch.cat([cls, pm], dim=1)
        return {"embeds": self._post(cls),
                "tokens": out.last_hidden_state[:, sl:].float().cpu().numpy()}


class Dinov3Anchor(BaseAnchor):
    """DINOv3-L/16 — 무언어 dense 앵커 (C2 fine 채널 ζ_f 기질, cowork §1/§2).

    native 규율 (W5 model-usage 감사 준수):
      • own norms / do_center_crop=False → 전체 프레임 정사각 no-crop resize.
      • registers 드롭: last_hidden = [CLS, R개 register, 패치…] → [:, 1+R:] (DINOv3-L R=4).
      • fp32 고정: DINOv3 patch 토큰은 fp16에서 NaN(W5 검증) → 항상 float32.
      • tokens = raw patch(ΔF 규약, SigLIP2 dense와 동일 — L2-norm 안 함).
      • has_text=False: 텍스트 쿼리는 메인 SigLIP2 앵커가 제공(dense 앵커는 patch만).

    해상도/그리드 (사전등록 §2.2 결정에 따라 config로 선택):
      • force_size=256 → 256/16 = 16×16 = 256 patch (SigLIP2-large-256 그리드 정합).
      • force_size=512, pool_to=16 → 32×32 인코딩 후 2×2 adaptive-avg-pool → 16×16 (FALLBACK:
        DINOv3 dense-optimum 보존 + 그리드 정합). 두 경로 모두 n_patch=256·dim=1024.
    cache_key = id(해상도+pool 판)/pre/raw → 해상도·pool별 dense 캐시 완전 분리.
    """
    has_text = False
    patch_dim = 1024

    def __init__(self, projection="pre", normalize=False, model_dir=None,
                 force_size=256, pool_to=None, pooled=None, crop=False):
        super().__init__("pre", normalize)             # joint 공간 없음 → pre 고정
        from transformers import AutoImageProcessor, AutoModel
        src = model_dir or "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp32 고정 (fp16 → NaN patch tokens, W5 검증). device 무관 항상 fp32.
        self.model = AutoModel.from_pretrained(src, dtype=torch.float32
                                               ).to(self.device).eval()
        self.n_reg = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.n_prefix = 1 + self.n_reg                 # CLS + register/storage tokens
        self.patch = self.model.config.patch_size
        self.force_size = int(force_size)
        self.pool_to = int(pool_to) if pool_to else None
        self.grid = self.force_size // self.patch      # 256/16=16 · 512/16=32
        assert self.grid * self.patch == self.force_size, \
            f"force_size {self.force_size} not divisible by patch {self.patch}"
        # no-crop 전체 프레임 정사각 resize (W5 DinoEncoder 미러 = 검증된 경로)
        self._size = {"height": self.force_size, "width": self.force_size}
        self.processor = AutoImageProcessor.from_pretrained(
            src, do_center_crop=False, size=self._size, do_resize=True)
        self.dim = self.model.config.hidden_size       # 1024
        self.dim_text = None
        out_grid = self.pool_to or self.grid
        self.n_patch = out_grid * out_grid             # 256 (grid-match, 두 경로 동일)
        self.id = f"dinov3-vitl16-{self.force_size}"
        if self.pool_to:
            self.id += f"-pool{self.pool_to}"          # 별도 dense 캐시 키
        # dual_stream(손목캠 변위 스트림): DINOv3는 native로 dense patch만 노출(embeds=None)
        #   → pooled 임베딩이 필요한 dual 손목 앵커용으로 CLS(또는 CLS⊕patch-mean) pooled를
        #   옵션 반환. pooled=None(기본)이면 embeds=None·id 불변 → 기존 dense 경로 비트 동형.
        assert pooled in (None, "cls", "clsmp"), pooled
        self.pooled = pooled
        if pooled:
            self.id += f"-{pooled}"                    # 별도 pooled 캐시 키 (dense와 분리)
            self.dim = self.model.config.hidden_size * (2 if pooled == "clsmp" else 1)
        # P-A crop arm: encode 직전 콜리그-EXACT 224 center-crop 후 native resize.
        # crop=False(기본) = 기존 no-crop 전체프레임 경로 비트 동형. id 접미사 = 캐시 분리.
        self.crop = bool(crop)
        if self.crop:
            self.id += "-crop224"

    @torch.no_grad()
    def encode_images(self, pil_images):
        import torch.nn.functional as F
        if self.crop:                                  # P-A: crop → native resize 순서
            pil_images = _crop224(list(pil_images))
        toks, embs = [], []
        for i in range(0, len(pil_images), 16):        # OOM 안전 서브배치 (512 fp32)
            inp = self.processor(images=pil_images[i:i + 16], return_tensors="pt",
                                 size=self._size).to(self.device)
            out = self.model(pixel_values=inp["pixel_values"].to(self.model.dtype))
            t = out.last_hidden_state[:, self.n_prefix:, :]     # drop CLS+regs → (B,P,1024)
            if self.pooled:                             # dual 손목 pooled: CLS(+patch-mean)
                cls = out.last_hidden_state[:, 0]
                if self.pooled == "clsmp":
                    cls = torch.cat([cls, t.mean(dim=1)], dim=1)
                embs.append(self._post(cls))
            if self.pool_to:                            # FALLBACK: g×g → pool_to×pool_to avg-pool
                B, _P, D = t.shape
                g = self.grid
                x = t.reshape(B, g, g, D).permute(0, 3, 1, 2)          # (B,D,g,g)
                x = F.adaptive_avg_pool2d(x, (self.pool_to, self.pool_to))
                t = x.permute(0, 2, 3, 1).reshape(B, self.pool_to * self.pool_to, D)
            toks.append(t.float().cpu().numpy().astype(np.float32))
        return {"embeds": np.concatenate(embs, 0) if self.pooled else None,
                "tokens": np.concatenate(toks, 0)}


class RadioAnchor(BaseAnchor):
    """C-RADIOv4-SO400M + SigLIP2 어댑터 — 언어 정렬 "리치 앵커" (F1, cowork 방향 2a).

    RADIO는 여러 교사(DINOv2/SigLIP2/SAM)를 한 백본에 증류한 agglomerative 모델.
    'siglip2-g' 어댑터의 **summary** 출력이 SigLIP2 텍스트 타워와 같은 공간에 정렬됨
    → 이 summary를 앵커 임베딩으로, 짝지어진 텍스트 인코더로 t2a/zero-shot 가능.

    검증된 torchhub API (README/hubconf, 2026-07 실측):
      m = torch.hub.load('NVlabs/RADIO', 'radio_model',
                         version='c-radio_v4-so400m', adaptor_names=['siglip2-g'],
                         progress=True, skip_validation=True)
      out = m(x)                              # x: (N,3,H,W), 값 [0,1]
      summary, features = out['siglip2-g']    # RadioOutput 네임드튜플 언팩
      # 텍스트: 어댑터가 open_clip 텍스트 타워를 노출
      tok = m.adaptors['siglip2-g'].tokenizer(texts)
      temb = m.adaptors['siglip2-g'].encode_text(tok, normalize=False)

    ⚠ SPEC과의 차이: 계획서는 adaptor_names='siglip' 이었으나 C-RADIOv4가 실제로
      노출하는 어댑터명은 'siglip2-g'(외 dino_v3/sam3). → 'siglip2-g'로 정정.

    입력 해상도(고정·문서화): FIX_RES=512. RADIO는 임의 해상도를 지원하지만 저/고해상도
      모드 차이가 있어 하나로 고정해야 함 — C-RADIOv4-SO400M/SigLIP2-g는 고해상도 증류
      기반이므로 512(고해상 모드)로 고정하고 get_nearest_supported_resolution로 스냅.
      center-crop 없음: 전체 프레임을 (R,R)로 직접 resize (시뮬 렌더 테두리 보존).
    전처리: RADIO 자체 conditioner가 [0,1] 입력을 mean0/std1로 정규화 → 우리는 픽셀을
      [0,1]로만 넘김 (make_preprocessor_external 호출 안 함).

    F3(dense/patch) 닫힘 — summary만. encode_images의 tokens는 항상 None.
    """
    id = "c-radio-v4-so400m"
    has_text = True
    patch_dim = None                            # dense 미사용 (F3 closed)
    FIX_RES = 512                               # 고정 입력 해상도 (위 주석 참조)

    def __init__(self, projection="joint", normalize=True, model_dir=None):
        if projection != "joint":
            raise ValueError("radio: projection=pre 미지원 (어댑터 정렬 공간만)")
        super().__init__(projection, normalize)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._adaptor_name = "siglip2-g"
        self.model = torch.hub.load(
            "NVlabs/RADIO", "radio_model", version="c-radio_v4-so400m",
            adaptor_names=[self._adaptor_name], progress=True, skip_validation=True,
            trust_repo=True,
        ).to(self.device).eval()
        self._adaptor = self.model.adaptors[self._adaptor_name]
        # 지원 해상도로 스냅 (min_resolution_step = patch*window 배수 제약)
        self.res = self.model.get_nearest_supported_resolution(self.FIX_RES, self.FIX_RES)[0]
        # summary/text 차원을 더미 forward로 확정 (스펙: 추측 금지, 실측)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.res, self.res, device=self.device)
            summary, _ = self.model(dummy)[self._adaptor_name]
            self.dim = int(summary.shape[-1])
            tok = self._adaptor.tokenizer(["x"]).to(self.device)
            self.dim_text = int(self._adaptor.encode_text(tok, normalize=False).shape[-1])
        self.save_tokens = False                # F3 닫힘 — 항상 None

    def _preprocess(self, pil_images):
        """center-crop 없이 전체 프레임을 (res,res)로 resize → [0,1] 텐서 (RADIO 규약)."""
        import torch.nn.functional as F
        px = []
        for im in pil_images:
            a = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
            a = F.interpolate(a.unsqueeze(0), size=(self.res, self.res),
                              mode="bilinear", align_corners=False, antialias=True)
            px.append(a)
        return torch.cat(px, 0).to(self.device)

    @torch.no_grad()
    def encode_images(self, pil_images):
        out = self.model(self._preprocess(pil_images))
        summary, _feat = out[self._adaptor_name]        # RadioOutput 언팩 (summary만 사용)
        return {"embeds": self._post(summary), "tokens": None}

    @torch.no_grad()
    def encode_texts(self, texts):
        tok = self._adaptor.tokenizer(list(texts)).to(self.device)
        emb = self._adaptor.encode_text(tok, normalize=False)   # 정규화는 _post에서 일괄
        return {"embeds": self._post(emb), "tokens": None}


class _Dinov2FusionBranch:
    """DINOv2-large pooled-global 브랜치 (외부 콜리그 SigLIP/src/core/dual_wrapper.py 레시피 미러).

    dual_wrapper.py:61-79 verbatim: raw AutoModel(facebook/dinov2-large) + pooler_output,
    L2정규화는 호출측(DualFusionAnchor.encode_images)이 수행(콜리그 F.normalize와 동형).

    전처리 (isolation-experiment 핵심 — "backbone vs protocol" 분리):
      • force_size=None → **DEFAULT AutoImageProcessor** (DINOv2 기본: 256 resize→224 center-crop).
                          = 콜리그 EXACT(1차 arm). dual_wrapper.py:62가 쓰는 바로 그 경로.
      • force_size=N    → do_center_crop=False, N×N no-crop resize (2차 arm: 예 256으로
                          SigLIP2-large-256 그리드 정합). config 한 줄(null→256)로 arm 전환.
    Dinov2Anchor 를 재사용하지 않은 이유: (1) Dinov2Anchor 기본이 do_center_crop=False·force-224
      ('nc' 판)라 콜리그의 기본(256→224 crop) EXACT 가 아님, (2) @256 matched-grid arm 을 만들
      force_size 노브가 없음, (3) 항상 dense 토큰을 계산/반환(pooled-only 융합엔 낭비)하고 자체
      normalize/id 규약을 가짐. 콜리그 레시피는 ~4줄이라 verbatim 미러가 가장 충실한 통제.
    """
    def __init__(self, model_dir=None, force_size=None):
        from transformers import AutoImageProcessor, AutoModel
        src = model_dir or "facebook/dinov2-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 콜리그와 동일: cuda면 fp16, cpu면 fp32 (dual_wrapper.py:42-44,61).
        self.model = AutoModel.from_pretrained(
            src, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.force_size = int(force_size) if force_size else None
        if self.force_size:                          # 2차 arm: no-crop N×N (그리드 정합)
            self._size = {"height": self.force_size, "width": self.force_size}
            self.processor = AutoImageProcessor.from_pretrained(
                src, do_center_crop=False, size=self._size, do_resize=True)
            self._id = f"dinov2_{self.force_size}nc"
        else:                                        # 1차 arm: 콜리그 EXACT = 기본 전처리
            self._size = None                        # size 인자 미전달 = dual_wrapper.py:62 동형
            self.processor = AutoImageProcessor.from_pretrained(src)
            self._id = "dinov2_def"
        self.dim = int(self.model.config.hidden_size)   # 1024


class DualFusionAnchor(BaseAnchor):
    """Arm A — 관측(anchor)-레벨 avg-fusion (외부 E3 dual_wrapper 공식, cowork §3-2).

    z = L2norm( (1-alpha)*L2norm(sig_pooled) + alpha*L2norm(dino_pooled) ),  alpha=0.5.
      • sig 브랜치 = SigLIP2-large-256 pooled (get_image_features, 1024-d, joint 공유공간).
      • dino 브랜치 = **dino_dir 로 백본 선택** (isolation 실험 "DINOv2 vs DINOv3"):
          - dino_dir 에 "dinov2" 포함 → _Dinov2FusionBranch (콜리그 dual_wrapper.py 레시피,
            pooled global 1024, 전처리는 force_size=None 이면 콜리그 EXACT / int면 @N matched).
          - 그 외 → Dinov3Anchor pooled CLS (1024-d, fp32 고정) — **기존 경로 byte-identical**.
      • 각 브랜치를 L2정규화 후 가중합, 합을 다시 L2정규화 ("두 unit의 평균은 unit이 아님"
        스케일 교정 — 외부 주석 dual_wrapper.py:9-11). dim 불변(1024) → 하류 차원 변경 없음.
    텍스트는 SigLIP2 텍스트타워로 위임(언어정렬 성분 보존; DINO 계열은 텍스트 타워 없음).
    외부 원본 공식: SigLIP/src/core/dual_wrapper.py:75-82 (verbatim, DINOv2 판).
    로딩/전처리/native 규율은 기존 Siglip2Anchor·Dinov3Anchor 내부를 재사용(중복 회피).
    """
    has_text = True
    patch_dim = None                                 # pooled-only fusion (dense 미노출)

    def __init__(self, projection="joint", normalize=True, model_dir=None,
                 siglip_dir=None, dino_dir=None, alpha=0.5, force_size=256,
                 crop="none"):
        if projection != "joint":
            raise ValueError("dualfusion: projection=pre 미지원 (SigLIP2 공유공간 위임)")
        super().__init__("joint", normalize)
        self.alpha = float(alpha)
        # P-A crop arm (§8): none(기본, 기존 비트 동형) / dino(DINO 브랜치만) / both(양 브랜치).
        #   sig 브랜치 crop 은 Siglip2Anchor 내부에서, dino 브랜치 crop 은 _dino_pooled 에서
        #   (브랜치 processor 를 직접 호출하는 경로라 융합측이 담당) 적용.
        assert crop in ("none", "dino", "both"), crop
        self.crop = crop
        # sig 브랜치는 normalize=True 고정 (공식의 L2norm(sig_pooled) — 외부 SiglipWrapper 동형).
        self._sig = Siglip2Anchor(projection="joint", normalize=True,
                                  model_dir=siglip_dir or model_dir
                                  or "google/siglip2-large-patch16-256",
                                  crop=(crop == "both"))
        # 백본 선택 = dino_dir. "dinov2" 포함 → DINOv2 콜리그 판, 그 외 → 기존 DINOv3(불변).
        dino_dir = dino_dir or "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self._is_dinov2 = "dinov2" in dino_dir.lower()
        if self._is_dinov2:
            # DINOv2 arm: force_size=None → 콜리그 EXACT(기본 전처리) / int(예 256) → matched arm.
            self._dino = _Dinov2FusionBranch(model_dir=dino_dir,
                                             force_size=force_size)
            dino_tag = self._dino._id                # dinov2_def / dinov2_{N}nc
        else:
            self._dino = Dinov3Anchor(projection="pre", normalize=False,
                                      model_dir=dino_dir, force_size=force_size)
            dino_tag = f"dinov3_{self._dino.force_size}"     # 기존과 동일 (256)
        # DINOv2 콜리그-기본 arm(force_size=None)은 processor 가 이미 256→224 center-crop
        # → crop 옵션과 결합하면 이중 crop (침묵 오염) → loud fail.
        if self.crop != "none" and self._is_dinov2 and self._dino.force_size is None:
            raise ValueError("crop: dinov2 콜리그-기본 arm 은 processor 가 이미 center-crop "
                             "— 이중 crop 금지 (force_size 지정 또는 crop: none)")
        self.device = self._sig.device
        assert self._sig.dim == self._dino.dim, \
            f"dualfusion: SigLIP dim {self._sig.dim} != DINO dim {self._dino.dim}"
        self.dim = self._sig.dim                      # avg → SigLIP dim 유지 (1024)
        self.dim_text = self._sig.dim_text            # 텍스트 = SigLIP2 (1024)
        # id 는 백본판별 접미사로 분기 → dinov2/dinov3 캐시 완전 분리.
        # DINOv3(dinov3_256)는 기존 문자열과 byte-identical.
        self.id = f"dualfusion-sig{self._sig.dim}-{dino_tag}-a{self.alpha}"
        if self.crop != "none":
            self.id += f"-crop224{self.crop}"         # P-A: crop arm 별 캐시 키 완전 분리

    def _dino_pooled(self, pil_images):
        """DINO pooled global (외부 dual_wrapper 의 dino.pooler_output 대응).

        HF 출력에 pooler_output 있으면 사용, 없으면 CLS=last_hidden[:,0]
        (Dinov2Anchor 주석 = HF 검증: DINO pooler == last_hidden[:,0]).
        DINOv3(fp32) 서브배치는 Dinov3Anchor.encode_images 규약 미러 — size=_size 항상 전달.
        DINOv2 1차 arm(_size=None)은 size 인자 미전달 = 콜리그 dual_wrapper.py:62 EXACT.
        P-A crop(dino|both): 브랜치 native 전처리 직전 콜리그-EXACT 224 center-crop.
        """
        if self.crop in ("dino", "both"):
            pil_images = _crop224(list(pil_images))
        d = self._dino
        outs = []
        for i in range(0, len(pil_images), 16):       # fp32 OOM 안전 서브배치
            kw = {} if getattr(d, "_size", None) is None else {"size": d._size}
            inp = d.processor(images=pil_images[i:i + 16], return_tensors="pt",
                              **kw).to(d.device)
            out = d.model(pixel_values=inp["pixel_values"].to(d.model.dtype))
            pooled = (out.pooler_output
                      if getattr(out, "pooler_output", None) is not None
                      else out.last_hidden_state[:, 0])
            outs.append(pooled.float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def encode_images(self, pil_images):
        import torch.nn.functional as F
        pil_images = list(pil_images)
        a = self._sig.encode_images(pil_images)["embeds"]       # (N,1024) L2-norm np.float32
        a = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        b = F.normalize(self._dino_pooled(pil_images), dim=-1)  # (N,1024) L2-norm
        assert b.shape[-1] == self.dim, f"dino dim {b.shape} != {self.dim}"
        z = F.normalize((1.0 - self.alpha) * a + self.alpha * b, dim=-1)   # unit, 1024-d
        return {"embeds": z.float().cpu().numpy(), "tokens": None}

    @torch.no_grad()
    def encode_texts(self, texts):
        """DINOv3는 텍스트 타워 없음 → 순수 SigLIP2 텍스트 pooler (1024-d, 미혼합)."""
        return self._sig.encode_texts(texts)


class DualConcatAnchor(DualFusionAnchor):
    """Arm B — no-mix concat fusion (cowork §3-2). 로딩/텍스트/pooled 는 DualFusionAnchor 재사용.

    z = concat([L2norm(sig_pooled), L2norm(dino_pooled)]) → 2048-d.
      • concat 은 재정규화하지 않음 (각 서브블록이 이미 unit; 2048 전역 재정규화는 두 기질의
        상대 스케일을 뭉개어 no-mix 취지를 훼손 → 미적용. avg판의 unit 스케일 교정과 목적이 다름).
      • 텍스트 = SigLIP2 서브블록(첫 1024-d)만 → 언어정렬은 SigLIP2 블록 대상 (§3-2).
    ⚠ dim_text(1024) != dim(2048): phase2 lang 토큰은 z-공간 토큰들과 torch.stack
       (train_phase2.py:299)으로 균일폭 결합됨 → 폭 불일치로 concat arm 에서 stack 실패.
       loss/토큰 코드는 여기서 수정하지 않음 — SigLIP2 서브블록 정렬 hook 필요 (report 플래그).
    """
    has_text = True
    patch_dim = None

    def __init__(self, projection="joint", normalize=False, model_dir=None,
                 siglip_dir=None, dino_dir=None, force_size=256, crop="none"):
        super().__init__(projection=projection, normalize=normalize,
                         model_dir=model_dir, siglip_dir=siglip_dir,
                         dino_dir=dino_dir, alpha=0.5, force_size=force_size,
                         crop=crop)
        self.dim = self._sig.dim + self._dino.dim     # 1024 + 1024 = 2048
        self.dim_text = self._sig.dim_text            # SigLIP2 서브블록 = 1024
        # id 백본판별 분기 (DualFusionAnchor 와 동형) — dinov3_256 은 기존과 byte-identical.
        dino_tag = self._dino._id if self._is_dinov2 else f"dinov3_{self._dino.force_size}"
        self.id = f"dualconcat-sig{self._sig.dim}-{dino_tag}"
        if self.crop != "none":                       # P-A: id 재구성 후 crop 접미사 재부착
            self.id += f"-crop224{self.crop}"

    @torch.no_grad()
    def encode_images(self, pil_images):
        import torch.nn.functional as F
        pil_images = list(pil_images)
        a = self._sig.encode_images(pil_images)["embeds"]       # (N,1024) unit np.float32
        a = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        b = F.normalize(self._dino_pooled(pil_images), dim=-1)  # (N,1024) unit
        z = torch.cat([a, b], dim=-1)                 # (N,2048) — 재정규화 안 함
        assert z.shape[-1] == self.dim, f"concat dim {z.shape} != {self.dim}"
        return {"embeds": z.float().cpu().numpy(), "tokens": None}


_REGISTRY = {"clip": ClipAnchor, "siglip2": Siglip2Anchor, "dinov2": Dinov2Anchor,
             "dinov3": Dinov3Anchor, "radio": RadioAnchor,
             "dualfusion": DualFusionAnchor, "dualconcat": DualConcatAnchor}


def get_anchor(cfg=None):
    """config의 anchor 섹션으로 앵커 선택. 섹션 없으면 기존과 동일한 ClipAnchor."""
    cfg = cfg or load_config()
    a = cfg.get("anchor") or {}
    name = a.get("name", "clip")
    if name not in _REGISTRY:
        raise KeyError(f"unknown anchor '{name}' (지원: {sorted(_REGISTRY)})")
    _pre = name in ("dinov2", "dinov3")               # 무언어 = joint 공간 없음 → pre 고정
    _raw = name in ("dinov3", "dualconcat")           # concat = 서브블록 unit, 전역 재정규화 안 함
    kwargs = {"projection": a.get("projection", "pre" if _pre else "joint"),
              "normalize": a.get("normalize", False if _raw else True)}
    # P-A crop arm (§4.2(1)/§8): none(기본=기존 비트 동형) | dino | both.
    #   융합 앵커: dino=DINO 브랜치만 / both=양 브랜치. 단독 앵커 해석은 동일 의미론 —
    #   siglip2 는 both 일 때만, dinov3 는 dino|both 에서 crop (dino=DINO 브랜치 전용 의미).
    _crop = a.get("crop", "none")
    if _crop not in ("none", "dino", "both"):
        raise ValueError(f"anchor.crop {_crop!r} (지원: none|dino|both)")
    if _crop != "none" and name not in ("siglip2", "dinov3", "dualfusion", "dualconcat"):
        raise ValueError(f"anchor.crop 은 siglip2/dinov3/dualfusion/dualconcat 전용 ({name})")
    if name == "clip":
        return ClipAnchor(**kwargs, cfg=cfg)
    if name == "siglip2":
        kwargs["crop"] = _crop == "both"
    if name == "dinov2":
        kwargs["pooled"] = a.get("pooled", "cls")
        kwargs["center_crop"] = a.get("center_crop", False)   # 기본 = no-crop (v2 보정)
    if name == "dinov3":                              # C2 dense: 해상도/pool은 §2.2 결정 반영
        kwargs["force_size"] = a.get("force_size", 256)
        kwargs["pool_to"] = a.get("pool_to")          # None=native grid / 16=512→16×16 fallback
        kwargs["pooled"] = a.get("pooled")            # dual 손목 앵커: cls|clsmp (None=dense 전용, 기존 동형)
        kwargs["crop"] = _crop in ("dino", "both")
    if name in ("dualfusion", "dualconcat"):          # 관측-레벨 융합 (cowork §3-2)
        kwargs["siglip_dir"] = a.get("siglip_dir")
        kwargs["dino_dir"] = a.get("dino_dir")
        kwargs["force_size"] = a.get("force_size", 256)
        kwargs["crop"] = _crop
        if name == "dualfusion":
            kwargs["alpha"] = a.get("alpha", 0.5)
    return _REGISTRY[name](**kwargs, model_dir=a.get("model_dir"))
