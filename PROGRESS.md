# PROGRESS — CLIP Embedding Action (F-시리즈)

CLIP 잠재공간 기반 액션 표현 연구의 개발 로그. 각 항목 = **무엇을 / 어떻게 / 결과**.
설계도: `DESIGN_fusion_dense_latent_action_v1.md`, 착수 지시: `KICKOFF.md`.
> 내부 규율 문서(예측 장부·검증 로그·NUMBER_CARD)는 비공개(gitignore). 본 파일은 팔로업용 공개 로그.

---

## 🔄 현재 진행 중
- **원격 폐루프 롤아웃** (`kist_a6000_ss` GPU8/9, `MUJOCO_GL=osmesa`): libero_spatial 20롤/태스크
  - GPU8 `correct` = clean 코드가 문서상 **~85% 재현하는지** end-to-end 검증
  - GPU9 `wrong`+`blank` = 언어사용 판별(§0.6 R2)
  - osmesa(CPU 렌더)라 수 시간 소요. 완료 시 SR 분석·기록 예정.

---

## 로그 (최신 순)

### 2026-07-09 · 폐루프 결과: 포트 재현 검증 + 언어사용 판별 (§0.6 R2)
- **무엇을**: clean 리팩터 코드의 폐루프 재현 검증 + "정책이 언어를 실제 쓰는가" 판별평가.
- **어떻게**: 원격 GPU(osmesa), libero_spatial 20롤/task·1시드. correct(정상)/wrong(다른 태스크 지시문)/blank(빈 문자열) 3모드 폐루프.
- **결과**: **correct 80.0%**(문서 81-85 밴드 내 → **포트 회귀 없음 검증**), **wrong 5.5%**, **blank 0.0%** → correct−wrong **−74.5pp**, correct−blank **−80.0pp**. **§0.6 R2 통과: 정책이 언어를 결정적으로 사용**(언어 토큰이 손목캠 −34.8pp보다 큰 레버). 신뢰성=방향적 확정 — **wrong가 핵심 증거**(유효 지시문 불일치의 능동적 오도), blank는 OOD 혼입 주의. 확정은 3시드 + LIBERO-Para. (코드 경로 감사=버그 없음.)

### 2026-07-09 · cowork 검토 통합 (외부 이론 파트너)
- **무엇을**: cowork 검토노트 반영 — 구현 감사·문헌 재검증·우선순위 재정렬. (소통은 `docs/` 폴더.)
- **어떻게**: **F2 재판정** "부분확정"→**"필요조건 통과·dense 미검증"**(patch-mean=pooling이라 dense 구조적 미테스트, clsmp≈cls=전역표현 이점). F3·wrong/blank **사전등록**, arXiv ID 보정 3 + 경쟁논문 6 반영, SigLIP2 crop 확인(no-crop=정상), paraphrase 언어취약성(task2 100%→11.7%)을 판별지표로 승급. 회신 `docs/COWORK_REPLY_2026-07-09.md`.
- **결과**: 신규성 재정의(action-grounding × 언어보존 × dense-latent-bottleneck 결합; DynaFLIP이 frozen SOTA라 경계 좁아짐). 조치 큐 확정: F3에 **DINOv2-registers**, F7에 **sim-렌더 confound**, **LIBERO-Para** 공개벤치 배선, F1 RADIO(C-RADIOv4 2601.17237). (상세 내부 규율문서는 비공개.)

### 2026-07-09 · 리포 정리 (팔로업 가능성)
- **무엇을**: 새 합류자가 헷갈리지 않게 문서 정리.
- **어떻게**: README에 **문서 맵**(README=사용법 / DESIGN=설계 / KICKOFF=실행 / PROGRESS=로그)
  + F-시리즈 옵션·디렉터리 갱신, `src/README.md` 동기화(anchor·motion_lang·diagnosis 추가,
  policy `mlp/cls/pma`→`mlp/flow` 오기 수정), KICKOFF에 내부문서 caveat. (전담 서브에이전트 감사 후 적용.)
- **결과**: 정착 파이프라인 vs 활성 F-시리즈 구분 명확화, 코드-문서 일치.

### 2026-07-09 · 원격 파이프라인 구축 + EGL 렌더 이슈 해결
- **무엇을**: 외부 GPU 서버(`kist_a6000_ss`, RTX 6000 Ada ×10)에서 실학습·폐루프 실행 환경 구축.
- **어떻게**: GitHub push → 원격 `/workspace/CLIP_ws` 클론, data/models 심링크(기존 스냅샷 재사용),
  `~/clip_ws` 재지정. 시스템 python(libero+mujoco3.3.2+torch cu124) 사용. GPU 8·9.
- **결과**: phase1·phase2 실학습 **성공**(체크포인트 저장). 단 폐루프에서 `MUJOCO_GL=egl` 크래시
  — 컨테이너에 nvidia EGL 부재(+device_id 충돌). **`MUJOCO_GL=osmesa`(CPU 렌더)로 수정 → 정상**
  (task0 렌더·성공 확인). 학습 파이프라인은 원격에서 검증됨.

### 2026-07-09 · SigLIP2 토크나이저 정합성 검증 (서브에이전트)
- **무엇을**: SigLIP2 로드 시 `bos/eos_token_id 49406/49407`(CLIP 토큰) 경고가 텍스트 경로를
  오염시키는지 (F1 언어비교 유효성).
- **어떻게**: 전담 서브에이전트가 로컬에서 토크나이저 클래스·토큰 id·텍스트 임베딩 판별력 검사.
- **결과**: **정상(CORRECT)**. 토크나이저 = GemmaTokenizer(vocab 256000, eos=1). 경고는 config.json의
  CLIP 잔재 필드가 정적 range 검증에만 걸린 것으로 토큰화·`get_text_features` 미사용. 임베딩
  결정론적·판별적(동일 cos≈1.0, 상이 0.59~0.66). **§3 무오염, 수정 불요.**

### 2026-07-09 · F2 — dense 디코더빌리티 프로브 (go/no-go)
- **무엇을**: dense 표현이 CLIP-pooled보다 액션을 더 잘 디코딩하는지(오프라인) — 전체 dense 가설 관문.
- **어떻게**: `src/diagnosis/f2_dense_probe.py`. 인코더별 상태조건부 `[Δ표현, z_t]` → GT action chunk를
  RidgeCV+얕은 MLP로 회귀, held-out R². z-score 표준화(차원 confound 제거). 3시드·60ep libero_spatial.
  (초판이 음의 R²로 퇴화 → 측정 감사 후 상태조건부+RidgeCV로 수정 = 유효 영역 복원.)
- **결과** (CLIP-pooled 대비 R² gap, 3시드 평균): **DINOv2-cls +0.145 / clsmp +0.151 (견고)**,
  SigLIP2 +0.048(불안정), fusion +0.113(<best-single). 판정 **✅부분**. 단서: (a) 오프라인≠폐루프
  (E-series에서 DINOv2 오프라인 우세가 폐루프 −21.8pp 패배), (b) clsmp≈cls → patch 특이적 아님
  (전역 CLS 이점, 진짜 dense는 F3 attention-pool 필요), (c) 차원 confound 배제. → **F3 진행 정당**.

### 2026-07-09 · wrong/blank-instruction 판별 하네스 (§0.6 R2)
- **무엇을**: 정책이 언어를 실제로 쓰는지 측정(표준 SR로는 안 보임).
- **어떻게**: `rollout_sim.py`에 `--instruction-mode {correct,wrong,blank}` 추가. wrong=다른 태스크
  지시문(순환 오프셋), blank=빈 문자열. 기본 correct는 불변.
- **결과**: 로직 로컬 검증(correct≠wrong, blank=""). 실행은 원격(현재 진행 중). 측정 = correct−wrong/−blank.

### 2026-07-09 · HY03 하이브리드 언어정렬 이식
- **무엇을**: phase1에 언어정렬(1급 불변식) 복원 — 모션문장 대조로 언어축 유지.
- **어떻게**: `networks.py` `DeltaAE`에 `align_mode{dz,direct,hybrid}` + `info_nce`(SupCon 다중양성,
  학습형 온도). hybrid = dz 손실 + λc·InfoNCE(모션문장 타깃, `motion_lang.py`). train_phase1 배선.
- **결과**: **dz 기본 경로 비트 동형**(신규 파라미터는 가드 안, 손실 합산식 불변) + hybrid --smoke 실행 확인.

### 2026-07-09 · 앵커 추상화 이식 (F1 전제)
- **무엇을**: 다중 백본(CLIP/SigLIP2/DINOv2) 공통 인터페이스.
- **어떻게**: `src/core/anchor.py`(get_anchor 레지스트리). train_phase1가 앵커 선택, `latent_dim=anchor.dim`,
  libero 캐시 하위호환 폴백(기본 CLIP=기존 평면 캐시).
- **결과**: anchor=clip 기본 **비트 동형** + 3앵커 로컬 shape 검증(CLIP 768/1024, SigLIP2 1152, DINOv2 1024/2048).

### 2026-07-09 · F0 — latent_dim 일반화 리팩터
- **무엇을**: `policy.py`의 `LATENT=768` 하드코딩 제거 + dense 캐시 경로 신설 (이후 전 단계 전제).
- **어떻게**: `latent_dim` 파라미터화, phase2/eval이 phase1 체크포인트에서 주입. `dense_embeddings()` 추가.
- **결과**: **비트 동형 게이트 PASS** (phase1 22/22 + phase2 51/51 tensor `torch.equal`). anchor_proj 층은 F1로 연기.

### 2026-07-08 · 워크스페이스 마이그레이션
- **무엇을**: 정리된 LIBERO 전용 코드베이스를 신규 레포로 이전.
- **어떻게**: `github.com/Seung-Sub/CLIP_Embedding_Action` 신규 생성, 단일 커밋(Seung-Sub 단독), data/models 심링크.
- **결과**: 실행 가능한 clean 워크스페이스 확립.

---

## 다음 (계획, KICKOFF 순)
- 롤아웃 결과 → 재현 검증(≈85%) + 언어사용 판정 → 기록
- **F3** dense obs 융합(학습 attention-pool, 폐루프+P6 게이트) · **F1** RADIO 앵커 head-to-head
- 이후 F4(학습형 latent action) · F5(통합) · F6(아키텍처) · F7(frozen vs LoRA)
