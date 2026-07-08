# 설계 지시서 v1 — 융합·고해상도 표현 + 학습형 latent action 확장

> **대상**: Claude Code(실행자). 코드베이스·`HANDOFF.md`·E-시리즈(`RESEARCH_PLAN_reexp_anchor_fusion_v1.md`) 숙지 전제.
> **작성**: cowork(이론·문헌 검증 파트너) / 2026-07-08. 본 문서는 **전체 그림 설계·지시**이며, 구현·평가·수치화는 실행자가 수행.
> **정본·규율**: 수치 정본 = `outputs/report/*.json`, 인용 = `NUMBER_CARD.md`, 프로토콜 = `docs/eval_protocol.md`(fp16 비결정성 → suite 평균 n≥500만 공식, <3pp 반복, 시드 스크리닝 후 승자 전량). 각 실험 **실행 전 `upgrade_ledger.md`에 예측 등록**, 실행 후 판정. 판정 문서는 cowork 검토 큐로.
> **관계**: 본 계획(F-시리즈)은 진행 중 **E-시리즈의 상위 확장**이다. E-시리즈(앵커 공정 재판정·DINOv2 obs 융합)의 결론을 전제로 하며, 중복 실행 금지(§0.4 의존성 참조).

---

## 0. 검증 결과 요약 (실행자 확인용)

### 0.1 코드베이스 이해 — 검증 완료
- **파이프라인**: phase1 `DeltaAE`(g: 액션청크+z_t→ζ≈Δz / h: Δz+z_t→액션 / 손실 align·recon·cycle), phase2 `FlowPolicy`(토큰 [z_{t−n}, z_t, g(A_past), (lang), (wrist)] → ζ̂; source∈{noise,past,vision}; Euler K스텝). `networks.py`·`policy.py` 확인.
- **결정적 제약**: 전 파이프라인이 **프레임당 단일 pooled 벡터**로 작동. `policy.py`에 `LATENT=768` **하드코딩**, phase1 config `latent_dim:768`. `DeltaAE`/`ChunkEncoder`/`ChunkDecoder`는 `latent_dim` 파라미터를 받으나 실질 768 고정.
- **patch 토큰 미사용**: `ClipWrapper.encode_images`가 vision tokens(257×1024)를 `save_token_vectors`로 캐시에 저장하나, **phase1/2 잠재 경로엔 미투입**("정책 사용은 이후 단계"). → dense 확장의 진입점이자, 아직 비어 있는 자리.
- **앵커 추상화**(`src/core/anchor.py`): `ClipAnchor`(768 joint / 1024 pre), `Siglip2Anchor`(1152, `get_image_features`=MAP head, Gemma 토크나이저), 공통 인터페이스 `encode_images`(embeds+optional tokens)·`encode_texts`·`has_text`·`projection`·`pooled`·`dim`. → **새 앵커 추가의 확장점.**
- **데이터**(`data/libero.py`): LIBERO, OSC 7-DoF 델타, agentview + optional eye_in_hand(wrist), 청크 0.8s/16스텝@20Hz, 지시문은 파일명 기반 CLIP 텍스트 임베딩 캐시.
- **실행 체계**: YAML config(`configs/phase1_libero*.yaml` 등) 구동.

### 0.2 모델 정보 재수집 — 정리 (근거: arXiv/공식)
| 모델 | 전역/summary | dense/공간 | 텍스트 | 해상도·전처리 | 이 프로젝트에서의 자리 |
|---|---|---|---|---|---|
| **CLIP ViT-L/14** | 768 joint(투영 CLS), L2-norm | 257×1024 vision tokens(약함, 대조 붕괴) | 있음(77 tok) | 224, 자체 transform | 현행 앵커(베이스라인) |
| **SigLIP2-so400m-384** | 1152, **MAP head**(`get_image_features`) | last_hidden_state patches(SigLIP1·CLIP보다 개선) | 있음(**sigmoid**, Gemma, ≤64) | 384, **resize(no-crop)** | 앵커 후보(VL) |
| **DINOv2(-reg)** | CLS(텍스트 미정렬) | patch 강함(공간) | 없음 | patch-14, resize | obs 공간 특징(E-시리즈) |
| **DINOv3**(2508.10104) | CLS | **Gram anchoring**로 고해상도 dense 붕괴 방지, 2D RoPE 고해상도, 최고 dense 품질 | **post-hoc 텍스트 정렬 옵션**(dino.txt류) | 고해상도(patch-16 계열), multi-res | obs 고해상도 dense / (텍스트 정렬 시) 앵커 후보 |
| **C-RADIOv4**(NVIDIA, 2026-02) | summary 토큰(**adaptor=siglip/clip → 언어정렬**) | dense spatial(any-res) | SigLIP 텍스트 어댑터 경유 | any-res/aspect, 자체 전처리, windowed attn | **단일 forward로 앵커(summary)+obs(dense) 동시** |

관련 방법론(설계 근거): **DINO-WM**(2411.04983 — frozen DINOv2 **patch** 공간에서 미래 patch 예측=dense dynamics, "표현/dynamics 분리", 접촉조작 강함), **LAPA**(VQ latent action)·**Genie**·**VLM-LAM**(2601.22714, promptable VL feature 타깃)·**JALA**(2602.21736, IDM latent 정렬)·**AM-RADIO/RADIOv2.5**(2312.06709/2412.07679, agglomerative distillation). 공통 교훈: **latent action은 raw Δ가 아니라 병목으로 학습**, dynamics는 **dense patch 공간**에서, 여러 인코더는 **증류/융합**으로 통합.

### 0.3 핵심 판단 (설계의 근거)
"CLIP 잠재공간 변위=액션" 아이디어는 옳으나, **단일 전역 벡터의 raw 변위**는 약한 인스턴스다(전역 대조 붕괴로 공간 정보 소실 / 제어·잡음 변화 얽힘 / 초구면 뺄셈 기하 부정합 / 자명해 위험). 정공법은 아이디어를 유지하되 **(I) 앵커 공간을 융합으로 풍부화, (II) 관측을 dense patch로, (III) raw Δ를 dense 위 학습형 latent action으로 승격**하는 것. 단, **"풍부함 ≠ 폐루프 개선"** — proprio 정보↑→SR↓(−28pp)·DINO 앵커 무익이 경고하므로, 모든 단계는 폐루프+P6 shortcut+자명해 프로브로 게이트한다.

### 0.4 E-시리즈와의 의존성 (중복 금지)
- **F1**(풍부한 앵커)는 E1/E2(앵커 공정 재판정)에 **RADIO/DINOv3-text 팔을 추가**하는 형태 — E2 프로토콜 재사용.
- **F3**(dense obs 융합)은 E3(DINOv2 mean-pool→patch attention)의 **일반화**(DINOv3/RADIO-spatial + SigLIP2-dense). E3가 미완이면 F3는 E3 위에서 확장.
- **F4/F5**(학습형 latent action·통합)는 E2 앵커 결론 확정 **이후** 착수.

### 0.5 왜 pooled 비교만으로는 부족한가 (검증 근거 + 핵심 함의)
**검증된 사실**(SigLIP2 논문 arXiv 2502.14786 및 후속 벤치마크): CLIP/SigLIP 대비 SigLIP2 이득은 **global/pooled 과제(zero-shot·retrieval)에서 작고**(동일 크기 ~2–3%p; 예 ImageNet zero-shot B/16@256 76.7→79.1), **dense·localization에서 큼**(세그멘테이션 최대 +5 mIoU; RefCOCO testA 70.1→86.2 ≈ +16%p). 기전도 명확: SigLIP2의 dense 이득은 **LocCa(un-pooled 표현에 cross-attention하는 캡션·referring 디코더) + self-distillation + masked prediction**에서 오며, **un-pooled patch에 산다**(pooling 시 소실). DINOv3의 강점(Gram anchoring 고해상도 dense)도 마찬가지로 patch에 있다.

**함의(이 연구에 직결)**: 현행 파이프라인은 프레임당 **pooled 전역 벡터의 변위**만 쓴다. 따라서:
- CLIP↔SigLIP2를 **pooled 앵커로 비교하면 ~2–3%p짜리 작은 global 차이만 포착하고 큰 dense·localization 이점은 구조적으로 버려진다.** → 이것이 E-시리즈의 **CLIP≈SigLIP2(무승부)의 유력한 원인**이다.
- **F1(pooled 앵커 교체)은 richer 백본에 대해 near-null test**다 — SigLIP2/RADIO-summary가 여기서 CLIP을 못 이겨도 "백본 열위"가 아니라 "pooled가 이점을 지움"으로 해석해야 한다(F1을 그렇게 판정).
- **richer 백본이 이 아이디어에 실제로 유리한지의 공정한 검증은 dense를 쓰는 F2(진단) → F3/F4에서만 가능**하다. **F4(dense 위 학습형 latent action)가 "각 모델에 맞는 새 latent space"의 실체**다 — 사용자가 지목한 "pooled로는 부족, 새 방법 필요"의 구현.
- **백본 계열별로 "변위/action code"의 형식을 달리한다**: global-VL(CLIP/SigLIP2 pooled)은 전역 벡터 차(제한적) / **dense(DINOv3·SigLIP2 patch·RADIO spatial)은 patch 필드 위 학습형 공간 latent action**(전역 뺄셈 아님, 필요 시 patch 대응/흐름 형태). **하나의 틀에 강제 금지.**

### 0.6 능동 검증: 전체 연구의 주의점 (검증 근거 + 설계 반영)
cowork가 지시 없이 스스로 식별해 **실제 논문으로 검증**한, v1에 아직 충분히 반영되지 않은 load-bearing 리스크. 각 항목은 설계 반영 지시를 동반한다.

**(R1) frozen prior의 근본 한계 = "semantic gap"(도메인 갭 아님).** VLM4VLA(arXiv 2601.03309): **실제 이미지로 학습해 sim 갭이 없어도, 비전 인코더를 동결하면 저수준 제어 성능이 크게 하락하고 인코더를 풀면 급등**. 원인은 도메인 갭이 아니라 **semantic gap** — 멀티모달 이해용 특징이 저수준 조작에 필요한 fine-grained 표현과 정렬돼 있지 않음. X-IL(2502.12330)도 frozen CLIP이 조작에서 약하고 fine-tune이 크게 낫다고 검증. 단 DINOv2 **patch**는 전이력이 높아 world model에 frozen으로 쓰임(DINO-WM). → **반영**: "frozen prior 활용" 전제가 성능 천장을 만들 수 있다. **frozen vs LoRA vs prior-preserving 적응(PriorVLA류 2605.10925)** 을 1급 절제(F7)로 추가 — "동결 제약이 얼마의 비용인지, prior를 지키며 제어 적합성을 회복 가능한지" 측정. LoRA는 언어·의미 prior를 일부 침식하므로 **반드시 언어 축과 함께 평가**.

**(R2) 표준 LIBERO는 언어를 사실상 테스트하지 않음.** LIBERO-PRO(2510.03827)/후속: **지시문을 지우거나 손상시켜도 모델이 무시하고 훈련된 vision-action 패턴으로 성공**하는 경우가 흔함. → **반영**: 본 연구의 핵심(언어 능력 보존)은 **표준 성공률로 검증 불가**. **잘못된/공백 지시 대조(wrong/blank instruction)** 를 추가해 "지시를 실제로 쓰는가"를 측정하고(성공률이 떨어져야 정상), 언어 이점은 **paraphrase·instruction-perturbation** 에서 평가(기존 `paraphrases.py` 확장). C8의 "언어 축=DA 선호"도 이 대조로 재확인.

**(R3) 표준 LIBERO는 포화·과적합 → 백본 판별력 낮음.** LIBERO-PRO/-X/-Plus: >90% 모델이 **경미한 위치·지시 변화에 붕괴**(over-fit), 표준 성공률이 모델 차이를 가림(예 LIBERO-X Level-1에서 평균 90→39%). → **반영**: §0.5(pooled near-null)와 결합하면 **표준 성공률로 CLIP/SigLIP2/DINOv3 비교 시 richer prior 이점이 이중으로 안 보인다**(pooled가 dense를 지우고 + 포화가 차이를 가림). **판별 슬라이스로 perturbation/OOD 평가(LIBERO-Plus/-PRO 또는 위치 perturbation)를 표준 성공률과 병행** — richer prior·dense의 일반화 이점은 여기서 드러나며, 이게 이 연구의 진짜 주장("더 나은 prior → 더 나은 일반화")을 검증하는 자리.

**(R4) 아이디어는 미세·접촉 조작에 약하고 + 손목캠이 심각하게 과소활용됨(코드·문헌 검증).** 시각 변위 Δz는 큰 팔 이동엔 크지만 grasp·insertion 같은 미세·접촉 동작엔 작아 인코딩이 어렵다(LAPA grasping 약점). **코드 확인**: 손목캠(`z_wrist`)은 phase1(액션 접지)·변위에 전혀 안 쓰이고 **현재 프레임 pooled 단일 토큰**으로만 정책 문맥에 들어가는데, 그 토큰을 빼면 **85.2→50.4%(−34.8pp)** — 가장 중요한 입력을 가장 얕게 쓰는 중. **검증된 원리(2507.17141)**: 액션 좌표계를 그 동작을 관찰하는 카메라 프레임에 맞추면 시각-액션 정렬이 개선된다 — **손목캠(eye-in-hand)↔EE/카메라 프레임 delta(미세·접촉에 강함), 3자뷰(base 고정)↔base 프레임 delta(대범위 이동에 강함)**. HoMeR(2506.01185)는 reaching=absolute/base·미세=relative/EE 하이브리드로 +29pp; camera-frame delta pose(2606.17846/2512.11218)는 "시각적으로 비슷한 동작=수치적으로 가까움"으로 액션을 관측공간에 정렬. → **반영(F3/F4에 1급)**: 사용자 제안대로 **agentview Δz = base-frame coarse action, 손목 Δz = EE-frame(camera-frame delta) fine action**으로 접지하는 **2-스트림 역할분리 latent action**; 그리퍼(open/close)는 손목 관찰성이 높으니 **별도 헤드** 고려; 미세 조작 구간 분해 지표(접촉 단계 성공/실패) 병행. **전면 구현은 F0–F2 기반 검증 후 F3/F4에서** 착수(우선순위는 그때 재평가).

**관련 선행(신규성·감사 반영)**: **DynaFLIP(2605.30350)** — LIBERO에서 frozen 인코더들을 **frame-transition embedding(인접 프레임쌍 융합)** 으로 비교하는, 우리와 **가장 가까운 선행**(단 인코더 fine-tune·확산정책). related work 최우선 대조 + 프로토콜(reusable-encoder frozen vs LoRA) 참고. **dino.txt(2412.16334)** — frozen DINO에 텍스트 타워를 LiT식(global+patch-avg concat) 정렬, open-vocab seg에서 SigLIP 능가 → **DINOv3-as-anchor의 텍스트 경로로 실재**(frozen 탓 텍스트 품질 caveat). **PriorVLA(2605.10925)** — prior 보존 적응(R1 중간 경로). 이 셋은 `related_competitors.md`에 추가. **카메라-프레임 액션 정렬(R4)**: 전신 조작 액션표현 비교(2507.17141), HoMeR 하이브리드 absolute/relative(2506.01185), camera-frame delta pose(2606.17846·2512.11218) — F3/F4의 2-스트림 역할분리 설계 근거로 인용.

---

## 1. 설계 원칙

**두 역할의 분리(H1 계승)**:
- **앵커(action code의 타깃 잠재)**: 언어정렬 global-semantic 필요 → CLIP / SigLIP2 / **RADIO-summary** / (텍스트 정렬된) DINOv3.
- **관측(정책 문맥)**: 공간 정밀도 필요 → **dense patch**(DINOv3 / RADIO-spatial / SigLIP2-dense). **전역 pooling 단일 토큰 금지**(shortcut 유발, E-시리즈 P6 교훈).

**세 개선 축**: (I) 풍부한 앵커, (II) 풍부한 관측, (III) **raw Δ → 학습형 dense latent action**. 값싼 것부터, 각 축을 **독립 절제**로.

**언어 정렬 = 1급 불변식 (모든 백본·단계에서 유지)**: C8 결과(폐루프는 Δz-접지 선호, **언어 축은 직접 정렬 선호**, 하이브리드 λ_c=0.3가 폐루프 손실 없이 언어 회복)와 phase2 `lang_token` 필수를 근거로, **하이브리드 정렬(Δz-접지 + 언어 직접 정렬) + phase2 언어 토큰**을 새 백본(SigLIP2/RADIO/DINOv3-text)에도 **반드시 이식**한다. 이유: 본 연구의 핵심 가치는 "사전학습 prior(특히 **언어**)를 잃지 않고 액션을 접지"이므로, 언어 정렬을 빼면 아이디어 자체가 훼손된다. 각 백본의 텍스트 경로(SigLIP2=자체 sigmoid 타워, RADIO=siglip 어댑터+SigLIP 텍스트, DINOv3=dino.txt류 정렬 변형)로 언어 정렬을 **반드시 구성**하고, 언어 축 지표(t2a·zero-shot·paraphrase 폐루프)를 매 비교에 병기한다. **주의**: 기본 phase1 config엔 언어 항이 없고 HY03 확장에 있으므로, 실행자는 그 하이브리드 정렬을 이월해야 한다. → F1/F4의 "선택적 언어정렬"을 **필수**로 격상.

---

## 2. 단계별 지시 (F0–F5)

### F0 — 활성화 리팩터 (비실험 · 선행 필수)
현행 768 고정을 일반화. **실험 아님 — 이후 전 단계의 전제.**
- **latent 차원 파라미터화**: `policy.py`의 `LATENT` 상수를 `latent_dim` 인자로. `FlowPolicy`/`MLPConcat`/`DeltaAE` 모두 `latent_dim` 주입받도록. 기존 CLIP-768 런과 **비트 동형** 회귀 테스트로 검증(수치 불변 확인).
- **앵커→latent 사영 옵션**: 앵커 pooled dim(예: SigLIP2 1152, DINOv3 1024, RADIO 가변)을 공통 `latent_dim`으로 사영하는 선택적 선형층(`anchor_proj`). config `model.anchor_proj: {enabled, out_dim}`. 없으면 `latent_dim=anchor.dim`.
- **dense 캐시 경로 신설**: 앵커에서 patch/spatial 토큰을 저장하는 캐시(`dense_cache_dir`). pooled 캐시와 **분리 키**(`{anchor}/{res}/pooled|dense`). 대용량이므로 옵션·태스크 서브셋부터.
- **회귀 게이트**: CLIP-768 확정 레시피가 리팩터 전후로 offline 지표·1시드 폐루프 **불변**이어야 병합.

### F1 — 풍부한 앵커: RADIO / DINOv3-text (off-the-shelf, 최저비용)
E1/E2에 팔 추가. **"통합 풍부 공간 > CLIP" 가설의 최소비용 검증.**
- **구현**:
  - `RadioAnchor`(`anchor.py`): torchhub `NVlabs/RADIO`, `model_version=c-radio_v4-so400m`(우선) / `-h`. `adaptor_name=siglip` → **summary(언어정렬)** 를 앵커 임베딩으로, **텍스트는 SigLIP2 텍스트 타워**(어댑터 짝) 경유. `has_text=True`, `dim`=summary 차원. dense spatial은 F3에서 사용(동일 forward에서 반환하도록 `tokens` 채움).
  - (선택) `DinoV3TextAnchor`: 텍스트 정렬 변형(dino.txt류)이 있으면 앵커 후보로, 없으면 F1에서 제외하고 F3 obs 전용.
- **실험**: E2 head-to-head에 팔 추가 — **CLIP vs SigLIP2 vs RADIO-summary**(+ 가능 시 DINOv3-text). E2 규약 그대로(각 native 레시피=E1, paired, 50roll×3seed, 차분 CI). RADIO 전처리·정규화는 §3 감사 준수.
- **사전등록 판정**: RADIO-summary가 최고 VL 앵커 대비 **차분 CI 분리 개선**이면 "통합 앵커 우위", 겹치면 "구별 불가"(현행 유지). offline 선정 금지(§0.3).
- **주의**: RADIO는 저/고해상도 모드 분리 이슈 있음 — 해상도 고정·명기. summary는 CLS류이므로 dynamics 적합성은 dense(F3) 대비 열세일 수 있음(가설). **§0.5대로 pooled 앵커 비교는 near-null test** — SigLIP2/RADIO가 CLIP을 못 이겨도 백본 열위로 결론 금지, dense(F2/F3/F4)에서 재검증. F1의 실효 가치는 "우열 판정"보다 **언어 정렬·인터페이스 검증 및 dense 확장의 발판**에 있음.

### F2 — dense 디코더빌리티 프로브 (초저비용 진단)
**dense 가설의 go/no-go.** flow 없이, 인코더별 **Δ(표현)→GT action 선형/얕은 프로브**로 R²·MAE 측정:
- 팔: CLIP-pooled / SigLIP2-pooled / DINOv3-patch(pooled 또는 소수 attention 토큰) / RADIO-spatial(pooled) / **융합(DINOv3+SigLIP2 patch)**.
- Δ 정의는 각 공간 native(§3). dense는 patch grid의 attention-pool 또는 flatten 후 프로브.
- **판정**: dense/융합이 CLIP-pooled 대비 R² 유의 상승이면 F3/F4 진행 근거. 상승 없으면 dense 가설 재검토(부정 결과도 기록). (E0/E1 진단 양식 재사용.)

### F3 — 풍부한 관측: dense patch 융합 (E3의 일반화)
E3(DINOv2 mean-pool→attention-pool)를 **DINOv3/RADIO-spatial + SigLIP2-dense**로 확장.
- **구현**:
  - 앵커/인코더가 patch 토큰(`tokens`)을 반환하도록(F0 dense 캐시). DINOv3(고해상도, register 변형)·RADIO-spatial·SigLIP2 last_hidden_state.
  - `ObsFusion`(E3에서 신설): 인코더별 공통 차원 사영 → **토큰축 concat**(OpenVLA/Prismatic식) → **K개 학습 쿼리 cross-attention(attention pooling)** → K obs 토큰을 `FlowPolicy` 조건 토큰에 append. `n_query` 기본 8. mean-pool 단일 토큰 **금지**.
  - 토큰 폭증 시 **pixel-unshuffle/토큰 축소**(RADIOv2.5식) 또는 저해상도 grid부터.
- **실험**: **앵커=E2 승자 고정**(융합 효과 격리). 팔: (a) obs 없음, (b) E3 meanpatch, (c) DINOv3-attnpool, (d) RADIO-spatial-attnpool, (e) DINOv3+SigLIP2 융합-attnpool, (f) **2-스트림 역할분리(R4): 손목 dense를 EE-frame 액션에 접지(camera-frame delta)하는 별도 obs/latent 스트림 + agentview base-frame**. (f)는 손목캠 −34.8pp·검증 원리(2507.17141)를 감안한 우선 팔. 스크리닝→승자 50roll×3seed.
- **사전등록 판정(게이트)**: 최고 dense 팔이 **(a) 대비 + (b) 대비 차분 CI 개선 AND P7 강건성 비열화**면 통과. **P6 shortcut 스크린 필수**(단독 토큰 MAE가 시각 토큰 이하이면 적신호 → causal confusion 위험, E-시리즈 교훈).

### F4 — 큰 아이디어: 학습형 dense latent action (Phase-1 재설계)
raw pooled Δz 타깃을 **융합 dense feature 위 학습형 공간 latent action**으로 교체. 현행 파이프라인의 **상위집합**(flow+decoder 유지). **이것이 §0.5의 "각 모델에 맞는 새 latent space"의 실체** — pooled가 지우던 SigLIP2/DINOv3의 dense·localization 이점을 action code가 실제로 활용하게 만드는 단계. 백본 계열별로 latent action 형식(patch attention 토큰 수·병목·대응 방식)을 달리 튜닝.
- **구현**:
  - **백본(frozen)**: F1 승자(RADIO 단일 forward) 또는 DINOv3-dense + SigLIP2 융합 → dense grid F_t + 언어/summary 채널.
  - **latent action encoder A_φ**(학습, 소형): (F_t, F_{t+k}) → 공간 latent action(소수 토큰/저차원 필드), cross-attention/Perceiver. 손실:
    - (i) **정보 병목**(VQ 또는 연속 KL) — 잡음 요인 필터(LAPA/Genie).
    - (ii) **역동역학 디코더빌리티**: h(latent, z_t)→A_fut L1 (JALA식 action grounding). — 기존 recon 자리 계승.
    - (iii) **잡음 불변성**: 시점·조명·distractor 증강 하에서 latent action 일관(consistency). — 제어 가능 변화 분리.
    - (iv) **언어정렬(필수, §1 불변식)**: pooled/summary 성분에 백본별 native 텍스트 손실(SigLIP2=**sigmoid**+native temp, CLIP/RADIO=InfoNCE/어댑터) — 전체 잠재를 텍스트 공간에 강제하지 않되 언어 축은 유지. λ_c는 HY03(0.3) 기준으로 시작해 백본별 재탐색.
  - `FlowPolicy`가 이 latent action을 예측, **frozen decoder h**가 로봇 액션 복원. `latent_dim`=latent action 차원(F0 일반화 활용).
- **사전등록 판정**: F1–F3 최고 대비 폐루프 **차분 CI 개선 AND 자명해 프로브 통과**. **자명해 프로브는 기존 C1(`encoder_state_cond=False`)·C0(`decoder_state_cond=False`) 기전을 재사용·확장**한다(신규 구현 불필요): F_t를 {제거/zeros/shuffle} 시 align·dec R²·a2z가 유의 하락해야 정상. 하락이 노이즈 이하이면 latent action이 상태 조건이 아니라 정적 타깃 프레임 인코딩임 → flag. 병목 유형(VQ vs 연속)·불변성 유무 절제.
- **주의**: 토큰·연산 폭증 → attention pooling·토큰 축소 필수. 오프라인 지표 상승은 채택 근거 아님(§0.3, 폐루프만).

### F5 — 통합 (조건부)
**앵커=F1 승자 + 관측=F3 승자 + 학습형 latent action=F4**, 각 게이트 통과 시에만.
- **실험**: 확정 아키텍처 50roll×3seed(LIBERO-Spatial) → 통과 시 suite 확장(Object/Goal)·G3 혼동행렬.
- **판정**: 각 단일 축 최고 대비 CI-비열화 이상이면 승격. 좌표를 NUMBER_CARD에 명기(π0 96.8 / OpenVLA 84.7 / OpenVLA-OFT 97.6 / LAPA 73.8 기준).

### F6 — 아키텍처 품질 축 (아이디어·백본 검증과 **분리**, 후행)
현행 g/h/policy는 단순 MLP·1D-CNN이다. 이는 **아이디어 검증엔 적절**(최소 기계로도 작동 실증)하나 성능 천장을 만든다. **핵심 규율: 아키텍처 최적화는 아이디어/백본 비교와 교란되면 안 되므로, 백본·앵커·latent-action 승자가 확정된 뒤에만 그 위에서 돌린다.** "아이디어가 된다"와 "아키텍처가 최적이다"를 절대 혼동하지 말 것.
- **후보 개선(각각 독립 절제)**:
  - **청크 인코더**: 시간축 mean-pool(순서 소실) → 소형 causal/attention 또는 순서보존 집계(청크 내 시간 구조 활용).
  - **phase1 디코더 h**: 결정론 MLP(같은 (Δz,z_t) 다봉 표현 불가) → (선택) 소형 생성형/구조화 디코더(ACT식 CVAE 등). 단 ζ가 특정 액션을 인코딩하므로 이득은 잔여 다봉성에 한정 — 폐루프로 검증.
  - **정책 v-net/ctx**: 잔차 MLP → 토큰 attention(dense obs와 결합 시 자연스러움, F3와 통합 가능).
  - **grid 세밀화**: 승자 근방 λ·용량·lr 미세 탐색(E1 native 레시피와 연계).
- **사전등록 판정**: 각 개선이 확정 아키텍처 대비 폐루프 **차분 CI 개선 AND P7 비열화**일 때만 채택. 개선 없으면 단순 버전 유지(단순성은 신규성 서술에 유리).
- **주의**: 이 축의 목적은 성능이지 아이디어 검증이 아니다. F1–F5 결론(아이디어·prior·언어 정렬의 타당성)은 이 축과 무관하게 성립해야 한다.

### F7 — frozen 제약 비용 측정: frozen vs LoRA vs prior-preserving (R1 대응)
**목적**: VLM4VLA/X-IL 검증(§0.6 R1)대로 frozen 인코더가 저수준 제어에 semantic gap을 가지므로, **"동결 제약이 얼마의 비용인지, prior를 지키며 제어 적합성을 회복 가능한지"** 를 측정. **아이디어 검증과 분리** — 승자 백본·앵커 확정 후 그 위에서.
- **팔**: (a) frozen(현행 전제) / (b) 인코더 **LoRA** 공동학습 / (c) **prior-preserving 적응**(PriorVLA류 2605.10925 — prior 보존하며 제어 적합성 회복).
- **필수 병행 지표**: LoRA는 언어·의미 prior를 침식할 수 있으므로 **언어 축(R2 wrong/blank instruction·paraphrase) + perturbation/OOD(R3)** 를 반드시 함께 측정. "제어 성공률↑ 그러나 언어·일반화↓"인지 확인.
- **사전등록 판정**: (b)/(c)가 (a) 대비 폐루프 **차분 CI 개선**이면 "frozen 제약이 유의미한 비용"이라는 결론(연구 서사에 중요). 단 **prior/언어 보존**이 핵심 주장이므로, prior 침식 없이 개선하는 (c)를 선호. 개선 없으면 frozen 정당화(오히려 강한 주장).
- **주의**: 이 축은 "frozen이 최선인가"라는 **연구 전제 자체의 검증**이다. 결과가 어느 쪽이든 논문 서사를 강화한다(frozen 정당화 or prior-preserving 적응의 필요성).

---

## 3. 모델 사용법 감사 체크리스트 (실행자 준수)
각 인코더는 **native 규약**으로. 위반 시 비교 오염(E-시리즈 center-crop·CLS 사고 재발 방지).

- **CLIP**(베이스라인): pooled 768 joint, 224, 자체 transform. (현행 유지)
- **SigLIP2**: `get_image_features`(=MAP head) pooled 1152 / 정렬 손실은 **sigmoid + native temperature**(InfoNCE 아님) / **Gemma 토크나이저**·자체 텍스트 타워 / **384 resize, center-crop 금지** / dense는 last_hidden_state patches.
- **DINOv3**: HF/torchhub 공식 로드 / **register 변형** / patch-크기 배수 입력 / **고해상도는 2D RoPE로 native 지원**(고해상도 dense가 강점) / 텍스트 조건은 dino.txt류 정렬 변형 or 외부 SigLIP2 텍스트 / center-crop 금지(resize).
- **RADIO/C-RADIOv4**: torchhub `NVlabs/RADIO`, `c-radio_v4-so400m|-h` / `adaptor_name=siglip` → summary(언어정렬)+spatial 동시 / **RADIO 자체 전처리**(임의 해상도 지원) / **저/고해상도 모드 분리 주의**(해상도 고정·명기) / VLM식 토큰 과다 시 pixel-unshuffle 축소 / summary=CLS류이므로 dynamics엔 dense 병행 권장.
- **공통**: head 입력 **학습셋 통계 z-score 표준화**(차원·스케일 confound 제거) / 앵커별 캐시 키 분리 / 프롬프트 규약 대칭 적용.

---

## 4. 위험과 판정 규율
- **풍부함 ≠ 폐루프 개선**: 정보 증가가 shortcut/causal-confusion으로 **역효과** 가능(proprio −28pp). → 전 단계 **폐루프 SR + P6 shortcut 스크린 + 자명해 프로브**로 게이트. offline 지표는 선정 근거 아님(§6.2 해리).
- **연산·토큰 폭증**: dense/고해상도 → attention pooling·pixel-unshuffle·저해상도부터.
- **차원·기하 confound**: head 입력 표준화, 각 공간 native 기하(구면=log-map/정규화, dense=raw+표준화).
- **통계력**: 우열 주장은 3시드+paired 차분 CI, 겹치면 "구별 불가". suite 평균만 공식.
- **판별 평가 병행(§0.6 R2·R3, 필수)**: 표준 성공률은 포화·과적합·언어무시로 백본/prior 차이를 가린다. 따라서 **매 핵심 비교에 (a) wrong/blank instruction 대조(언어 사용 여부), (b) paraphrase/instruction-perturbation(언어 이점), (c) 위치 perturbation·OOD(LIBERO-Plus/-PRO)** 를 표준 성공률과 **함께** 보고. richer prior·dense·언어 정렬의 실효는 주로 이 판별 슬라이스에서 드러난다.
- **중복 금지**: F1⊂E1/E2, F3⊃E3. E-시리즈 결론 확정 후 F4/F5.

---

## 5. 실행 순서 · 산출물
1. **F0**(리팩터, 게이트: 회귀 불변) →
2. **F2**(dense 프로브, 저비용 go/no-go) 및 **F1**(RADIO 앵커, E2에 팔 추가) 병렬 →
3. **F3**(dense obs 융합, E3 확장) →
4. **F4**(학습형 latent action, E2 결론 후) →
5. **F5**(통합, 조건부) →
6. **F6**(아키텍처 품질, 승자 확정 후·아이디어 검증과 분리) ·
7. **F7**(frozen 제약 비용: frozen vs LoRA vs prior-preserving, 승자 확정 후). **주의**: F2와 함께 **판별 평가(§0.6 R2·R3: wrong/blank instruction·perturbation/OOD)** 를 조기에 세팅 — 표준 성공률만으로는 이후 비교가 백본 차이를 못 본다.

| 단계 | 산출물 |
|---|---|
| F0 | 리팩터 PR + `f0_regression.json`(불변 검증) |
| F1 | `f1_anchor_radio.json` + E2 판정 갱신 (**언어 축 지표 병기 필수**) |
| F2 | `f2_dense_probe.json` (+ 판별 평가 하네스 세팅) |
| F3 | `f3_obs_fusion.json` + P6/P7 |
| F4 | `f4_latent_action.json` + 자명해 프로브(C0/C1 재사용) |
| F5 | `f5_integrated.json` + suite 확장 + **perturbation/OOD 슬라이스** |
| F6 | `f6_arch_*.json` (개선별, 폐루프 차분 CI) |
| F7 | `f7_frozen_vs_lora.json` (언어·OOD 병행 지표 포함) |

각 단계 실행 전 `upgrade_ledger.md` 예측 등록, 실행 후 적중/반증. 사이클마다 `verification_log.md`. F1·F4·F5 판정은 cowork 큐로. 공표 수치는 NUMBER_CARD 정본.

---

*요약: 아이디어(사전학습 prior의 표현 변화에 액션을 의미적으로 접지)를 유지하되 "단일 전역 raw 변위"를 (I) 융합 앵커(RADIO/DINOv3-text) (II) dense 관측 (III) dense 위 학습형 latent action으로 승격. **언어 정렬은 전 단계 1급 불변식**. **검증된 전체 주의점(§0.6)**: frozen prior의 semantic gap(→F7 frozen vs LoRA vs prior-preserving), 표준 LIBERO의 언어무시·포화(→판별 평가 R2·R3 필수), 미세·접촉 조작 약점(→손목캠·근접뷰). 값싼 순서(F0→F2/F1→F3→F4→F5), 아키텍처·frozen 축(F6/F7)은 승자 확정 후. 폐루프+판별 평가 게이트, E-시리즈 무중복, 자명해 프로브는 C0/C1 재사용. 가장 빠른 신호는 F1·F2, 가장 큰 잠재력은 F4, 전제 검증은 F7.*

---

## 부록 D — 기존 CLIP 구현 감사 요약 (실행자 참고)
본 설계의 전제가 된, 현행 CLIP 코드베이스 감사 결론(cowork):
- **아이디어는 충실히 구현됨**(align=액션→Δz 접지 / recon·cycle=디코딩·일관성), 손실이 실문헌(VITA·FLD·A2A)에 매핑. **ARM-AE(정렬 제거 시 −7.4pp)가 CLIP 접지의 실효 기여를 증명** → 아이디어 유효.
- **잠정·한계(버그 아님)**: (1) 거친 그리드 → 근소 비교 잠정(E1이 처리). (2) 단순 MLP/CNN·시간 mean-pool·결정론 h → 검증엔 적절하나 성능 천장(F6이 분리 처리). (3) `align↔recon` 절충으로 ζ가 순수 Δz에서 이탈(dec R² 0.68 vs cycle 0.88) — **CLIP-전역-Δz의 액션 lossy성**을 반영, F-설계의 근본 동기. (4) 초구면 chord 기하 미세 부정합.
- **핵심**: 현행은 **타당한 proof-of-concept**이며 천장은 CLIP-전역-Δz가 설정. F-설계는 "동결 prior 접지 + 언어 유지"라는 검증된 아키텍처를 유지한 채 **더 나은 prior(공간·고해상도)·dense·학습형 latent action**으로 확장하는 정공법.
