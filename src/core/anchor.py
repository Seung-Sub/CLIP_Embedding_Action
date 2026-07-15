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

    def __init__(self, projection="joint", normalize=True, model_dir=None):
        if projection != "joint":
            raise ValueError("siglip2: projection=pre 미지원 (공유 공간 헤드 일체형)")
        super().__init__(projection, normalize)
        from transformers import AutoModel, AutoProcessor
        src = model_dir or "google/siglip2-so400m-patch14-384"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            src, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(src)
        self.dim = self.model.config.vision_config.hidden_size    # 1152
        self.dim_text = self.dim
        self.save_tokens = False                  # E3에서 True로 (패치 토큰 반환)

    @staticmethod
    def _tensor(out):
        """transformers 버전에 따라 get_*_features가 텐서 또는 출력 객체 반환."""
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def encode_images(self, pil_images):
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
                 force_size=256, pool_to=None):
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

    @torch.no_grad()
    def encode_images(self, pil_images):
        import torch.nn.functional as F
        toks = []
        for i in range(0, len(pil_images), 16):        # OOM 안전 서브배치 (512 fp32)
            inp = self.processor(images=pil_images[i:i + 16], return_tensors="pt",
                                 size=self._size).to(self.device)
            out = self.model(pixel_values=inp["pixel_values"].to(self.model.dtype))
            t = out.last_hidden_state[:, self.n_prefix:, :]     # drop CLS+regs → (B,P,1024)
            if self.pool_to:                            # FALLBACK: g×g → pool_to×pool_to avg-pool
                B, _P, D = t.shape
                g = self.grid
                x = t.reshape(B, g, g, D).permute(0, 3, 1, 2)          # (B,D,g,g)
                x = F.adaptive_avg_pool2d(x, (self.pool_to, self.pool_to))
                t = x.permute(0, 2, 3, 1).reshape(B, self.pool_to * self.pool_to, D)
            toks.append(t.float().cpu().numpy().astype(np.float32))
        return {"embeds": None, "tokens": np.concatenate(toks, 0)}


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


_REGISTRY = {"clip": ClipAnchor, "siglip2": Siglip2Anchor, "dinov2": Dinov2Anchor,
             "dinov3": Dinov3Anchor, "radio": RadioAnchor}


def get_anchor(cfg=None):
    """config의 anchor 섹션으로 앵커 선택. 섹션 없으면 기존과 동일한 ClipAnchor."""
    cfg = cfg or load_config()
    a = cfg.get("anchor") or {}
    name = a.get("name", "clip")
    if name not in _REGISTRY:
        raise KeyError(f"unknown anchor '{name}' (지원: {sorted(_REGISTRY)})")
    _pre = name in ("dinov2", "dinov3")               # 무언어 = joint 공간 없음 → pre 고정
    kwargs = {"projection": a.get("projection", "pre" if _pre else "joint"),
              "normalize": a.get("normalize", False if name == "dinov3" else True)}
    if name == "clip":
        return ClipAnchor(**kwargs, cfg=cfg)
    if name == "dinov2":
        kwargs["pooled"] = a.get("pooled", "cls")
        kwargs["center_crop"] = a.get("center_crop", False)   # 기본 = no-crop (v2 보정)
    if name == "dinov3":                              # C2 dense: 해상도/pool은 §2.2 결정 반영
        kwargs["force_size"] = a.get("force_size", 256)
        kwargs["pool_to"] = a.get("pool_to")          # None=native grid / 16=512→16×16 fallback
    return _REGISTRY[name](**kwargs, model_dir=a.get("model_dir"))
