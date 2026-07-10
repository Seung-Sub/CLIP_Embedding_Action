# PROGRESS — CLIP Embedding Action (F-시리즈)

CLIP 잠재공간 기반 액션 표현 연구의 개발 로그. 각 항목 = **무엇을 / 어떻게 / 결과**.
설계도: `DESIGN_fusion_dense_latent_action_v1.md`, 착수 지시: `KICKOFF.md`.
> 내부 규율 문서(예측 장부·검증 로그·NUMBER_CARD)는 비공개(gitignore). 본 파일은 팔로업용 공개 로그.

---

## 🔄 현재 진행 중
- **1c 씬내 타깃-스왑 프로브** — 같은 씬에 실재하는 다른 태스크 타깃으로 지시문 스왑(cross-task
  wrong보다 인과적: 새 타깃이 씬에 실제 존재). 실행 중, 예비 결과 태스크별 이질적(task5 instructed
  100% / task8 75% vs task0·task3 대부분 "neither"=실패) → **최종 suite mean 대기.**
- (완료) **방향1 언어사용 굳히기**: 1a 3시드(correct−wrong +75.8pp)·1d-goal(+88.5pp)·1b(LIBERO-Para) — 아래 로그.
- (완료) **F3 dense-obs = 음성 종결**(a 50.0 > b 31.5 > c 15.5, 게이트 실패) → "richer obs 조건화가 폐루프 악화". F6 불요.
- (완료) **anchor 폐루프 경로 일반화**(commit 7ad1950) → 비-CLIP 앵커(SigLIP2/RADIO) 폐루프 head-to-head(F1) 언블록.

---

## 로그 (최신 순)

### 2026-07-10 · 1c: 씬내 타깃-스왑 프로브 (진행 중 · 예비)
- **무엇을**: 지시문을 **같은 씬에 실재하는 다른 태스크의 타깃**으로 스왑 — cross-task wrong보다 인과적
  (새 지시 타깃이 씬에 실제 존재하므로, 새 타깃으로 가면 구성적 접지 / 그냥 실패면 태스크선택자).
- **어떻게**: SWAP 맵 + 3-way **instructed/orig/neither** 검출기(env `_eval_predicate 'on'`), LIBERO-spatial. 실행 중.
  판별 평가(correct/wrong) 자체는 **우리 발명이 아님** — 동시대 벤치 **LIBERO-CF(2602.17659)·ICBench(2603.06001)**가
  "language-ignoring" VLA 실패 모드를 확립(기존 VLA `correct−wrong` ≤ **+19.4pp**). 우리 frozen-Δz 정책은
  개입 없이 통과(1a **+75.8** / 1d-goal **+88.5pp**) → 외부 벤치가 헤드라인을 **강화**.
- **결과 (예비, 최종 mean 대기)**: 태스크별 **이질적** — task5 instructed **100%**, task8 instructed **75%**
  (→ 구성적 접지) vs task0·task3 대부분 **"neither"(실패)**. 예비 판독 = **PARTIAL·태스크/위치 의존 구성적 접지**.
  **⚠ IN-PROGRESS — suite 최종 mean 대기, 예비값이며 판정 아님.**

### 2026-07-10 · anchor 일반화: 폐루프 경로에 get_anchor(cfg) (commit 7ad1950)
- **무엇을**: 폐루프 앵커 head-to-head(F1)를 언블록 — 비-CLIP 앵커(SigLIP2/RADIO)를 **전체 폐루프 파이프라인**에 흐르게.
- **어떻게**: phase2 학습 + sim/dataset 롤아웃이 `ClipWrapper()` 하드코딩 → **`get_anchor(cfg)`**로 교체(이전엔
  phase1+obs-fusion만 앵커 일반화, 폐루프는 CLIP 고정). 추가 수정: **지시문(언어) 임베딩 캐시를 앵커 `cache_key`로 키잉**
  (CLIP 768d vs SigLIP2 1152d가 아니면 충돌). **CLIP 기본 경로 비트동형 보존.**
- **결과**: CLIP 기본 비트동형 유지, 비-CLIP 앵커가 이제 폐루프까지 흐름 → **F1(RADIO/SigLIP2 substrate) 폐루프
  head-to-head를 판별 하네스와 함께 실행 가능**하게 언블록. 다음: 원격 여유 시 2b 언어축 head-to-head.

### 2026-07-10 · 1b: LIBERO-Para 객체 vs 동작 어휘 (부분적 의미 이해)
- **무엇을**: 언어 사용이 표면형(문자열 암기) vs 의미(패러프레이즈 강건)인지 분해.
- **어떻게**: LIBERO-Goal 정책, LIBERO-Para(HF, MIT) 패러프레이즈 객체-어휘/동작-어휘, 20롤/task. baseline goal correct 88.5%.
- **결과**: object-어휘 **51.0%**(−37.5pp) / action-어휘 **61.0%**(−27.5pp), Δ차 10pp(사전등록 표면형 임계 15pp **미만**). **판정 = 부분적 의미 이해**: 패러프레이즈에서 51-61% 유지(wrong=0%와 대비 → **의미 접지, 순수 암기 아님**), 단 **객체 명사가 최약축**(표면형 잔여, LIBERO-Para VLA 범위 내). correct 88.5 → paraphrase 51-61 → wrong 0 그라디언트가 (부분적) 진짜 접지 지지.

### 2026-07-10 · 1d-goal: 언어사용 일반성 (goal correct−wrong +88.5pp)
- **무엇을**: 언어사용(1a)이 goal suite에도 일반화되는가.
- **어떻게**: 신규 LIBERO-Goal 정책(phase1-goal dec 0.754 / phase2-goal R² 0.768), goal correct/wrong/blank 폐루프(20롤/task).
- **결과**: goal correct **88.5%** / wrong **0.0%** / blank **0.0%** → **correct−wrong +88.5pp** (spatial +75.8pp보다 강함). goal은 초기 씬 고정·언어가 유일 태스크 단서라 의존 더 큼 → 틀린/빈 지시문에 완전 전멸(0%). **언어사용 일반성 확정**(사전등록 sub "spatial 최대 델타"는 반증 — goal이 더 큼; 핵심 예측은 오히려 강하게 확정). 다음: 1b(객체 vs 동작 어휘) 실행 중.

### 2026-07-10 · 1a: 언어사용 3시드 확정 (correct−wrong +75.8pp)
- **무엇을**: −74.5pp(정책이 언어를 실제 사용)를 3시드로 확정 (헤드라인).
- **어떻게**: full-data phase2 s0/s1/s2(동결 full phase1 공유), correct/wrong 폐루프(20롤/task, osmesa, GPU 8/9 샤드).
- **결과**: correct−wrong = **s0 +76.5 / s1 +76.5 / s2 +74.5pp → 평균 +75.8pp**, 3시드 전부 CI로 0 분리(사전등록 적중 ✅). correct 82.7% / wrong 6.8%(3시드 평균). **언어보존·사용 헤드라인 확정.** blank는 OOD 보조. 다음: 1c(씬내 타깃스왑) + LIBERO-Goal 정책으로 1b(LIBERO-Para 객체/동작)+1d(goal 일반성).

### 2026-07-10 · F1 2a: RadioAnchor 구현 (Direction 2, 병행)
- **무엇을**: 언어정렬 richer 앵커 후보 RADIO — F3로 관측축 닫혀 이제 **타깃/앵커 축**(이기는 축).
- **어떻게**: `anchor.py`에 RadioAnchor = C-RADIOv4-SO400M(torchhub) + **`siglip2-g` 어댑터**(스펙의 `siglip`은 미존재 → 자료조사로 정정) summary=언어정렬 임베딩 + SigLIP2-g 텍스트 타워. dim 1536, res 512 whole-frame(no-crop), dense 불요(§F3 닫힘). 서브에이전트 구현+로컬 테스트, 실행자 검증.
- **결과**: dim 1536, 이미지·텍스트 동일 공간(cos 정상), get_anchor 등록, 기존 앵커 불변, deps(timm/open_clip) 설치. 다음(원격 여유 시): **2b 언어축 head-to-head**(CLIP vs SigLIP2 vs RADIO) — 1차 산출은 SR 아닌 언어축(§0.5).

### 2026-07-10 · F3 폐루프 판정: dense-obs **음성** (게이트 실패)
- **무엇을**: dense obs 융합이 폐루프 SR을 개선하는가 — 진짜 dense go/no-go.
- **어떻게**: 동일 120ep 공정 subset·동결 full phase1(dec 0.682) 공유, 3 arm 폐루프(20롤/task, osmesa): a(no-obs)/b(mean-patch)/c(DINOv2-reg attnpool).
- **결과**: **a 50.0% > b 31.5%(−18.5pp) > c 15.5%(−34.5pp)**. dense-obs가 폐루프를 해침 — 표현력 클수록 더(단조). 게이트(c가 a·b 양쪽 이겨야) **실패**. 단일 mean 토큰(b)조차 −18.5pp → dense **정보 자체가 해로운 shortcut**(토큰수/ctx 아티팩트 아님). 오프라인(a>b>c)과 방향 일치, 폐루프가 harm 증폭. **"풍부한 정보 추가→폐루프 악화" 패턴 3번째 확증**(proprio −28 · DINOv2앵커 −21.8 · dense-obs). 서사 변경 → cowork escalate, 다음 방향 PI/cowork 판단.

### 2026-07-09 · F3 통합: obs 융합 → phase2 (Task 2+4)
- **무엇을**: dense obs 토큰을 정책(phase2) 학습에 통합.
- **어떻게**: `libero.build_policy_samples`가 obs 앵커 dense patch를 subset materialize(캐시키 분리), `train_phase2`가 `module.obs` 게이트 하에 ObsFusion 빌드 + obs 토큰 K개 토큰열 뒤 append + 옵티마이저/체크포인트 통합. 신규 config `phase2_libero_obsc.yaml`(arm c). *(정밀 스펙+하드 게이트로 서브에이전트 구현, 실행자 독립 검증.)*
- **결과**: **게이트 통과** — (A) no-obs `--smoke` val_parts가 불변값과 **완전 일치**(비트동형, 독립 재검증), (B) obs arm(DINOv2-reg) 빌드+학습+저장 OK(124.16M→133.60M, n_tokens 5→13). 설계결정: full dense(~24GB) 비현실 → F3 초기는 동일 subset 공정비교, full은 lazy-loading 후속. 다음: rollout 통합 → arm b/e config → 원격 subset 폐루프 비교.

### 2026-07-09 · F3 착수: dense obs 융합 기반 (앵커 + ObsFusion)
- **무엇을**: F3(진짜 dense go/no-go) 구현 착수 — 계획(`docs/F3_PLAN.md`) 후 독립 기반 2개.
- **어떻게**: (1) `src/core/anchor.py`에 DINOv2-registers 변형(레지스터/CLS 제거→patch-only tokens, `-reg` 캐시키; 기존 앵커·기본 `dinov2-large` 경로 불변), (2) `src/models/obs_fusion.py` 신규 — 인코더별 patch→공통차원 사영→K=8 학습쿼리 cross-attention→K개 obs 토큰(768d), mean/pixel-unshuffle 지원. 둘 다 서브에이전트 구현+단위테스트, 실행자 diff 검토.
- **결과**: DINOv2-reg tokens `(N,256,1024)` 검증(261→[:,5:]로 patch만) / ObsFusion 4종 shape 테스트 통과. **no-obs 기본 경로 불변**(비트동형 보존). 다음: phase2/rollout 통합(no-obs 비트동형 게이트) + config(a/b/c/e) + P6/P7 포팅 → 원격 arm×3시드 폐루프.

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
