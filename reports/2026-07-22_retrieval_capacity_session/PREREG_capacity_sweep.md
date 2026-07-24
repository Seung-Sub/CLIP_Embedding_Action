# PREREG — g/h 용량(capacity) 스윕 v1 (2026-07-22, 오프라인 전용 사전등록)

*작성: 구현 에이전트. 목적: PI 의 잔존 의심 — "phase1 g/h 가 용량 부족이라 변위 통화의
상한이 낮게 잡혀 있는 것 아닌가" — 를 데이터로 판정한다. 근거: `src/models/networks.py`
(ChunkEncoder/ChunkDecoder hidden 폭), `src/training/train_phase1.py`(지표 산출부),
FOLLOWUP_experiments.md §10(Phase-A "복잡화 일관 무익" 프라이어), AUDIT §IV6(offline≠SR).
**mtime 규약**: 본 파일은 어떤 스윕 결과 JSON/ckpt 보다 먼저 기록된다 — 판정 기준의
사후 수정 금지.*

---

## 0. 결정 요약

| 항목 | 값 |
|---|---|
| 질문 | phase1 DeltaAE 의 g(ChunkEncoder)·h(ChunkDecoder) hidden 폭(현행 512)이 병목인가? |
| 팔 | **8개** = {g-스윕, h-스윕} × 폭 {0.5, 1, 2, 4}× (hidden {256, 512, 1024, 2048}) — 한쪽을 움직일 때 다른 쪽은 512 고정 (귀속 분리) |
| 기질 | **large256-single** (`phase1_libero_siglip2_large256.yaml` 완전 상속, seed 0, 캐시 전량 재사용 — 인코딩 0회) |
| 범위 | **오프라인 전용** (phase1 refit + probe). 폐루프는 §4-B 게이트 통과 시에만 별도 사전등록으로 개설 |
| 코드 | `model.hidden_g`/`model.hidden_h` 노브 신설 (`networks.py DeltaAE`, 기본 None = hidden 상속 = **비트 동형**); phase2/rollout 재구성 배선 완료 (`train_phase2.py`/`rollout_dataset.py` `p1["model"].get(...)`) |
| 비용 | phase1 학습 ~30–60분/런 × 8 = **4–8 GPU-h** + probe(CPU/GPU 수 분/ckpt). 인코딩·캐시 0 |

왜 g/h 를 따로 스윕하는가: 의심의 두 하위가설이 다르다 — (i) **g 병목**: 액션청크→Δz
매핑 용량 부족이면 align cos·retrieval 이 폭에 반응해야 한다; (ii) **h 병목**: 잠재
1024(large256)를 hidden 512 로 좁혀 디코딩하는 병목이면 dec/cycle R² 가 반응해야 한다.
합산 스윕(hidden 단일 노브)은 어느 쪽이 움직였는지 귀속 불가.

주: `cap_g1` 과 `cap_h1` 은 아키텍처가 동일한 1× 재적합(둘 다 512/512, seed 0) — 결정론
환경이면 두 결과가 일치해야 하며, 이 일치 자체가 재현성 체크다. GPU 창이 빠듯하면 h1 을
생략하고 g1 결과를 h-곡선의 1× 점으로 공유해도 됨(사전 허용, 7런).

## 1. 배경 — 무엇이 이 셀을 정당화하는가

- Phase-A 프라이어는 "**구조** 복잡화 무익"이지 "**용량**"을 스윕한 적은 없다. hidden=512
  는 CLIP-768 레시피에서 그대로 이월된 값이고, large256 기질에선 latent 가 1024 로
  넓어졌는데 g/h hidden 은 512 그대로다 — h 는 in 2048(=[ζ;z_t]) → 512 → 112 로 좁아지는
  깔때기. PI 의심이 겨냥하는 지점이 정확히 여기다.
- 반대 프라이어: dec R² 는 이미 높은 편이고(레포 관례상 phase1 은 병목으로 지목된 적
  없음), offline≠SR(IV6) — 오프라인 R² 가 올라도 폐루프가 따라온다는 보장은 없다.
  그래서 본 셀은 **오프라인 선별**이고, 승격 조건을 §4 에 수치로 못박는다.

## 2. 팔 정의 (config 8개, 전부 작성 완료)

| config | hidden_g | hidden_h | checkpoint |
|---|---|---|---|
| `phase1_libero_large256_cap_g05.yaml` | 256 | (512) | `checkpoints/capsweep/large256_cap_g05.pt` |
| `phase1_libero_large256_cap_g1.yaml` | 512 | (512) | `.../large256_cap_g1.pt` |
| `phase1_libero_large256_cap_g2.yaml` | 1024 | (512) | `.../large256_cap_g2.pt` |
| `phase1_libero_large256_cap_g4.yaml` | 2048 | (512) | `.../large256_cap_g4.pt` |
| `phase1_libero_large256_cap_h05.yaml` | (512) | 256 | `.../large256_cap_h05.pt` |
| `phase1_libero_large256_cap_h1.yaml` | (512) | 512 | `.../large256_cap_h1.pt` |
| `phase1_libero_large256_cap_h2.yaml` | (512) | 1024 | `.../large256_cap_h2.pt` |
| `phase1_libero_large256_cap_h4.yaml` | (512) | 2048 | `.../large256_cap_h4.pt` |

hyper 는 전 팔 동일(batch 256, lr 1e-4, cosine, 50ep, early-stop 20, seed 0) — lr 를
폭에 맞춰 조정하지 **않는다**(단순 비교 우선; 4× 팔이 lr 민감으로 진동하면 그 사실
자체를 기록하고 lr 재스윕은 별도 결정).

## 3. 지표 (전부 기존 산출물 재사용 + probe 1개)

| 지표 | 출처 | 역할 |
|---|---|---|
| dec R² (`decoder_r2`) | train_phase1 표준 출력/ckpt metrics | **주 판정축** — h(Δz→a) 복구 용량 |
| cycle R² (`cycle_r2`) | 〃 | **주 판정축** — h(g(a)) 왕복 (phase2 디코딩 경로) |
| align cos (`align_cos`) | 〃 | g 매핑 품질 (g-스윕 반응축) |
| retrieval a2z/z2a top-1/5 | 〃 | g 매핑 판별력 (g-스윕 반응축) |
| **h-Jacobian eff-rank** | `scripts/probe_h_jacobian.py` (신규, 읽기 전용) | ∂h/∂ζ 의 유효 사용 차원 — "용량이 늘면 h 가 ζ 를 더 넓게 쓰는가" |

## 4. 사전등록 판정 (본 파일 mtime < 결과 mtime 의무)

주 판정축 = dec R² **와** cycle R² (1×-팔 값 기준, 각 스윕 곡선별로 판정):

- **A. 무죄 (capacity acquitted)** — 전 팔이 1× 대비 **±0.01 이내** (양 주 축 모두):
  용량 의심 기각. 셀 폐쇄, 폐루프 미개설. 논문엔 "g/h 폭 0.5–4× 스윕 flat" 한 줄 +
  Phase-A 프라이어 보강 증거로 수납.
- **B. 의심 지지 (suspicion supported)** — 폭에 **단조 증가**하고 (4×) − (0.5×) ≥
  **+0.03** (주 축 중 하나 이상): PI 의심 지지 → **폐루프 셀 개설** (별도 사전등록:
  승자 폭으로 phase2 재학습(matched 하이퍼) + matchedbase paired 20roll × 2모드,
  언어 공동기준 c−w ≥ +70 상속). 오프라인 상승은 개설 조건이지 승리 선언이 아님(IV6).
- **C. 중간/비단조** — 0.01 < Δ < 0.03 또는 비단조: 유보. seed 1개 추가(+8런 아님,
  경계 팔 2개만 재학습) 후 재판정; 여전히 애매하면 "약한 신호, 폐루프 미개설" 로 기록.
- **보조 해석 (게이트 아님, 사전 서약)**:
  - dec R² flat + eff-rank 단조 증가 → "여유 용량은 생기나 과제가 요구하지 않음" —
    무죄 서사의 기전 보강.
  - g-스윕에서 align cos/retrieval 만 오르고 R² flat → "매핑은 좋아지나 디코딩 상한
    불변" — 병목 위치를 h 쪽으로 지목하는 단서 (h-스윕 곡선과 교차 확인).
  - 어떤 팔이든 1× 대비 **하락** ≥0.01 → 과용량/최적화 문제 기록 (숨기지 않음).

## 5. 실행 (GPU 창 열리면 — 현재 W-A 확증 점유로 대기)

```bash
# clipx env, 캐시 박스. 순차 8런 (또는 GPU 2장이면 g/h 팔 병렬)
for a in g h; do for w in 05 1 2 4; do
  python src/training/train_phase1.py \
    --config configs/phase1_libero_large256_cap_${a}${w}.yaml
done; done

# probe (학습 종료 후, CPU 가능)
for a in g h; do for w in 05 1 2 4; do
  python scripts/probe_h_jacobian.py \
    --config configs/phase1_libero_large256_cap_${a}${w}.yaml
done; done
# 곡선 집계: ckpt metrics(decoder_r2/cycle_r2/align_cos/retrieval) +
#   outputs/capsweep/hjac_*.json (h_jac_effrank_mean)
```

## 6. 불변식·정직 섹션

- **byte-identity**: `hidden_g`/`hidden_h` 미설정 시 DeltaAE 생성 인자·순서·RNG 소비
  불변 (스모크 `scratchpad/test_pb_langselpool_smoke.py` HEAD 대조 (F) 단계에서 검증).
  기존 config/ckpt 전부 무영향.
- **하위 배선**: 스윕 ckpt 를 나중에 phase2/rollout 이 소비할 수 있도록
  `train_phase2.py`·`rollout_dataset.py` 의 DeltaAE 재구성에 `p1["model"].get("hidden_g"/
  "hidden_h")` 배선 완료 — 구 ckpt 는 키 부재 → None (비트 동형).
- **최강 반론**: "offline R² 는 SR 과 무상관(IV6) — 스윕이 이겨도 무의미". 응답: 그래서
  본 셀은 폐루프를 **열지 말지의 게이트**로만 쓰고(§4-B), flat 이면 GPU 4–8h 로 PI
  의심을 영구 종결하는 정보를 산다 — 지는 쪽도 산출물이 있는 실험만 연다는 레포 규율
  그대로.
- 결과 기록: 각 런 ckpt metrics + wandb(run_name=`large256_cap_*`) + probe JSON.
  본 문서는 결과 후 수정 금지(추가는 append "판정" 섹션으로만).
