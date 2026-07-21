# BRIEF — wrist 설계 입력 통합 (디자이너용 단일 근거 문서)

*작성 2026-07-18, 증거 통합 에이전트. 목적: wrist-cam 관련 내부 지식 전부를 수치+아티팩트 경로로
한 장에 고정 — 설계자는 이 문서만으로 재유도 없이 착수 가능. 신뢰 태그: EXACT(1차 아티팩트 존재) /
approx(문서 기재·부분 아티팩트) / UNVERIFIED(1차 근거 미회수).*

---

## 1. 확정 수치표

| # | 수치 | 값 | 아티팩트 | 태그 |
|---|---|---|---|---|
| 1 | wrist 조건토큰 절제 (구 CLIP-768 레짐, 5rep×10task×20롤) | 85.2 → 50.4% = **−34.8pp** | `experiments/baseline_5rep.jsonl`(83/85/86/86/86)·`experiments/wrist_excluded_5rep.jsonl`(51/50/50.5/47.5/53), 해설 `experiments/README.md:177`, 원출처 `DESIGN_fusion_dense_latent_action_v1.md:57`(§R4), 팔 config `configs/phase2_libero_nowrist.yaml` | **EXACT** (단, CLIP-768 구레짐 — 현 large256/concat 기질에서 미재측) |
| 2 | dual ζ_wrist zero-ablation ΔR²(A−B) | **+0.179** (0.6626→0.4832; GO 게이트 0.02의 9배) | `outputs/week0_probes/w0_p3_wrist.json` (per-dim R² 포함), 해설 `docs/WEEK0_probe_results.md` §6 | **EXACT** |
| 3 | 동 절제 oracle(GT Δz) Δ | +0.101 (0.7757→0.6748); wrist 단독 C=0.578, main 단독 C′=0.581 | 같은 JSON | **EXACT** |
| 4 | z_wrist 상태 셔플(D) | −0.009뿐 → 기여는 상태가 아닌 **변위 ζ_wrist 자체** | 같은 JSON | **EXACT** |
| 5 | oracle gripper-dim: main=0에서도 0.890 | 파지 채널은 wrist 변위만으로 거의 완전 디코딩 | 같은 JSON `Cp_oracle_main_zero.per_dim_r2[6]` | **EXACT** |
| 6 | ego-motion 프로브 E1 (액션→Δz_wrist ridge R²) | **+0.177** (게이트 0.5 기각; ctrl 액션→Δz_main +0.115) | 같은 JSON `probe3_ego_motion` | **EXACT** (선형 한정; MLP판·E2 proprio·R1 잔차 미실행) |
| 7 | Δz 스케일: main std **0.1346** / wrist std **0.0208** (6.5×) | phase1 dual 학습 로그 print | `outputs/remote_recovered/2026-07-18/kist_a6000_ss/outputs/logs/wrist.log:52` | **EXACT** (로그 회수됨; JSON 아티팩트는 여전히 없음) |
| 8 | dual phase1 오프라인: h R² **+0.738**, cycle 0.877, align cos main **+0.649** / wrist **+0.749** | wandb summary + 평가 블록 | `wrist.log:34-37,63-65`; ckpt `.../kist_a6000_ss` 회수분(원격 `/workspace/CLIP_ws/checkpoints/phase1_libero_dualstream_wrist.pt`, `CKPT_INVENTORY.txt:38`) | **EXACT** |
| 9 | dual phase2 오프라인: action R² **+0.663**, grip 93.5%, n_val 5502, 141.08M params, n_tokens=6, flow_dim=2048, **x0_std=0.0724(단일 스칼라)** | `wrist.log:96-125` (x0_std는 :115) | **EXACT** |
| 10 | dual 폐루프 부분판독 3회 | 66.7 / 76.2 / 80.6% — 최장 판독 **64/84=76.2%**, Wilson 95%CI **[66.1, 84.0]**, vs 85% p≈0.10 **NS** | `FOLLOWUP_experiments.md` §11, `docs/AUDIT_negative_results_2026-07-18.md` §4(d); 로그 부분회수: `roll_wrist_correct_t0..t3.log` = t0 14/20, t1 16/20, t2 14/16(부분), t3 11/13(부분) = 55/69 | approx (판독 3회 자체의 원로그는 부분만; 회수 로그 상 55/69=79.7%는 별도 창) |
| 11 | matched baseline(large256-single) 롤아웃 | **한 번도 안 돌았음** — 20개 전부 즉사: `FileNotFoundError .../checkpoints/phase2_libero_siglip2_large256.pt` (실체는 `checkpoints/grid/` 하위, `CKPT_INVENTORY.txt:44`) | `.../logs/mroll_base_{correct,wrong}_t0..t9.log`, 스크립트 `.../logs/wrist_matched.sh` | **EXACT** (경로 버그 확정) |
| 12 | 비교 기준 base | large256-single 85–88 / concat **97.5** / avg **91.5**(언어 +74) / no-aug baseline 85.0 | `FOLLOWUP_experiments.md` §4·§10; 감사상 **10/10 UNTRACED**(`AUDIT...md` §5) | UNVERIFIED (문서 기재; 로컬 1차 아티팩트 부재) |
| 13 | h Jacobian eff-rank (∂h/∂ζ) | PR 3.9–5.1 / latent 768–2048, 기질 불문 (콜리그 ≈5.5/1024 재현) | `outputs/week0_probes/w0_p2_jacobian.json` | **EXACT** |
| 14 | 콜리그 wrist 신호 강도 | grasp-프로브 AUROC: wrist 0.886–0.928 ≫ main 0.755–0.810; Jacobian 귀속 wrist 16.2–17.5%; mask-wrist→4%; wrist-scale↑ 노브 90.8(t4 60→86) | `SigLIP/report/S2_grasp_probe.md`, `SigLIP/report/attribution.md`, `SigLIP/report/post93_results.md:26` | approx (그들 프로토콜 — 절대비교 불가) |

**주의(감사 §5)**: SR 헤드라인 전반이 로컬 UNTRACED — 새 설계의 모든 롤아웃은 §4의 provenance 규율 필수.

---

## 2. 현행 구현 지도 (wrist 텐서 흐름, file:line)

### 2A. 단일 스트림 — wrist = 조건 토큰 1개 (현 best-base 경로, −34.8pp의 주체)
- **phase1**: wrist **완전 미사용** (DeltaAE는 agentview만).
- **데이터**: `src/data/libero.py:325-331` — `wrist_camera`(eye_in_hand_rgb) 프레임을 **main 앵커(clip)로 인코딩**한 z_wrist(t_cur) 1개 (aug 시 variants). 즉 현 레짐이면 SigLIP2-raw(대응 normalize 설정 그대로), 구 레짐이면 CLIP.
- **phase2**: `src/training/train_phase2.py:449-458` (`module.wrist_token` 플래그, 6번째 배열 분리) → 토큰열 `[z_prev, z_cur, g(A_past), lang, wrist]`, `:540` n_tokens 산정. 좁은 토큰은 z 폭으로 zero-pad.
- **rollout**: `src/eval_libero/rollout_sim.py:185-190`(encode_wrist — 비dual이면 main clip 사용), `:282-286`(토큰 추가; S1b cond_dim 슬라이스 `:283-284`), `:293-295`(zero-pad). 매 재계획 스텝마다 현재 프레임 재인코딩.

### 2B. dual-stream — wrist = 추론 변위 스트림 (Phase-B, INCONCLUSIVE)
- **앵커**: main=SigLIP2-large256 raw(normalize=false, cache_key `siglip2-so400m/joint/raw`) / wrist=DINOv3-L/16 pooled-CLS **normalize=true 단위벡터**(cache_key `dinov3-vitl16-256-cls/pre/norm`) — `configs/phase1_libero_dualstream_wrist.yaml:30-41`, **⚠SCALE 자기표기 `:13-15`**, 캐시 분리 `:16-18`.
- **phase1** `src/training/train_phase1.py:58-236` (run_dual): 삼중쌍 로드 `src/data/libero.py:319-324`; Δz std **print만** `:110-111`; `DualDeltaAE`(`src/models/networks.py:281-345`) = g_main·g_wrist(각 ChunkEncoder) + 단일 h(ChunkDecoder, in=dim_cat·2=4096, **입구 단일 LayerNorm `networks.py:50`**). 손실 `:331-345`: 0.5·(align_main+align_wrist)+0.5·recon+0.25·cycle — align은 스트림별 독립이나 **동일가중 원-스케일 MSE+cos**.
- **phase2** `src/training/train_phase2.py:55-253` (run_dual): 토큰 6개 `[zp_m, zc_m, a_emb, zwp, zwc, lang]` 각 2048로 zero-pad(`:168-173`); flow 타깃=concat ζ(`:178`); **x0_std = concat ζ 전체 단일 스칼라 `:155-159`** (실측 0.0724); wm cos 손실 concat(`:185-187`); 디코딩 frozen `ae.decode`(`:184`).
- **rollout** `src/eval_libero/rollout_sim.py:121`(anchor_wrist 로드), `:185-190`(dual이면 DINOv3로 인코딩), `:246-248`(zw_hist prev/cur), `:256-274`(dual 토큰·디코딩), `:335-336`(재계획마다 갱신). **주의**: dual은 단일 경로의 SigLIP2-wrist 토큰이 **사라지고** DINOv3-CLS prev/cur 2토큰으로 **교체**됨(= Phase-B 이중 변경의 ②).
- **byte-identity**: `dual_stream` 플래그 부재 시 단일 경로 완전 무변경 — `train_phase1.py:258-259`, `train_phase2.py:311-312`.

---

## 3. 감사 확정 결함 체크리스트 (AUDIT §4 + WEEK0 + 로그 실사)

| 결함 | 위치 | 수리 난이도 |
|---|---|---|
| ① 스트림 스케일 불일치 6.5×(0.1346/0.0208) 방치 — 스트림별 ζ 표준화 없음 | config ⚠주석 `phase1_libero_dualstream_wrist.yaml:13-15`; std는 `train_phase1.py:110-111` print만 | 쉬움 (dz_std buffer 등록 + z-score) |
| ② concat ζ 전체 **단일 스칼라 x0_std**(0.0724) — 저분산 wrist 블록의 x0가 main 스케일로 오염 | `train_phase2.py:155-159` | 쉬움 (per-dim/블록 상수, broadcast ~3줄) |
| ③ h 입구 **단일 LayerNorm**이 저분산 wrist 블록 감쇠 | `networks.py:50` (ChunkDecoder) | 중간 (① 표준화가 선행되면 사실상 해소) |
| ④ align "수렴" 착시 — cos 항 스케일 불변 + 동일가중 MSE(스케일²∝ main 지배). loss 항별 grad-norm 로깅 부재 | `networks.py:324-345`; 로그 `wrist.log:22-28` | 쉬움 (epoch별 항·grad-norm wandb) |
| ⑤ 이중 변경 confound: 변위 스트림 추가 + 조건 기질 교체(SigLIP2 1토큰→DINOv3-CLS 2토큰) 동시 | `rollout_sim.py:256-274` vs `:275-286` | 쉬움 (SigLIP2-wrist 조건토큰 유지 변형 팔) |
| ⑥ 통계 미성립: 결정론 롤아웃의 부분판독 3회=동일 에피소드 열의 절단 창(비독립); 커버리지 편향(task0–3+4ep); **matched baseline 실행 0회(경로버그, §1 #11)** | `AUDIT...md` §4(d); `mroll_base_*.log` | 중간 (retry-supervisor 200ep 완주 + grid/ 경로 수정 + paired CI) |
| ⑦ DINOv3-CLS-as-wrist 미검증 — 그리퍼가 화면 대부분인 wrist 뷰에서 CLS는 준상수 전역 요약 개연 | `docs/DESIGN_wrist_v2.md` §4c | 중간 (4c: 16×16 패치→2×2 pool 4토큰, param-0) |
| ⑧ 아티팩트 위생(과거): 단일 txt 덮어쓰기·per-episode 기록 부재 → UNTRACED 10/10 | `AUDIT...md` §5 | **해소됨** — `--run-tag` provenance 신설(§4) |

---

## 4. 신설계 제약·불변식

- **byte-identity**: 신기능은 플래그 뒤, 플래그 부재 시 기존 경로 비트 동형 (`dual_stream`·`wrist_token`·`cond_anchor` 전례). smoke + 기존 config 무변경.
- **no-aug 클린 밴드**: 융합/wrist 계열 평가는 no-aug 고정 (C2 감사: aug=−13pp 폐루프 confound, `FOLLOWUP...md` §3.3). dual 경로는 aug 자체를 assert 금지(`train_phase2.py:96`).
- **캐시 규율**: `anchor.cache_key`로 서브디렉터리 자동 분리(main `siglip2-so400m/joint/raw` / wrist `dinov3-vitl16-256-cls/pre/norm`) — 새 앵커/전처리는 **반드시 새 cache_key** (충돌=오염).
- **GPU**: 원격 A6000 **48GB**(17롤 동시실행 near-OOM 전례 `PROGRESS.md:51`; grid-token dense 인코딩 OOM 사망 전례 §10). 주의: 2026-07-18 현재 원격 박스 **CUDA 자체 불가**(cgroup, `docs/WEEK0_probe_results.md` §0) — GPU 복구가 모든 학습 셀의 선행 조건.
- **osmesa 불안정**: 확률적 침묵 세그폴트(ep 7–84), RAM-OOM 아님, EGL 불가(컨테이너) — `FOLLOWUP...md:161`, commit 0842fb1. **워크어라운드 = 태스크 단위 retry-supervisor(≤3회)**; 롤아웃 설계는 부분사(死)를 전제할 것.
- **provenance(신설, 사용 의무)**: `rollout_sim.py --run-tag` → `outputs/eval/runs/<tag>/episodes.jsonl` per-episode fsync 기록(`rollout_sim.py:96-114, :351-`) — ckpt 태그를 run_tag에 포함해 UNTRACED 재발 방지.
- **평가 프로토콜**: 폐루프 SR만 심판·사전등록(upgrade_ledger)·우세 주장은 3시드+paired bootstrap CI·이중 기준 **SR SIG>0 AND correct−wrong ≥ +70pp** (KICKOFF 불변식, DESIGN_wrist_v2 공통 사전등록).
- **suite 천장**: spatial concat 97.5 → 기전 검정은 **large256-single 기반(헤드룸 ~15pp)**, 승자만 concat-base/타 suite 확전(`DESIGN_wrist_v2.md` Stage 2 주의).
- **삽입점 프라이어**: 관측/조건화 = 유일 양성(S1, −34.8pp), 코드/타깃·디코더·앵커-only = 일관 무효(`FOLLOWUP...md` §3–7, §10) — 신설계는 실패 기전과의 차별화 명시 의무.

---

## 5. 설계자가 답해야 할 열린 질문

1. **손실 배치**: align_wrist를 어디에? (표준화 공간 MSE vs 잔차(ego-제거) 타깃 vs 조건화-only로 align 자체 폐지). P-zg1/zg3 적신호(align의 65–70%가 상태 성분, `WEEK0...md` §1–2)와 정합하게.
2. **스트림별 인코더**: wrist에 DINOv3-CLS 유지? SigLIP2 병기(c0)? 패치 2×2-pool 4토큰(4c)? — E1=0.177로 ego-지배 기각이므로 §4a(EE-frame) 선행게이트 미충족, **4c가 1순위**(`DESIGN_wrist_v2.md` §3).
3. **스케일 처리**: 스트림별 z-score의 위치(phase1 buffer vs phase2 x0_std per-block vs 양쪽) — P1-1 예측 "표준화로 ΔR²(A−B) ≥1.5×"의 검정 설계 포함.
4. **phase2 주입점**: wrist를 조건 토큰(검증 양성)만으로? flow 타깃(코드-측, 일관 무효 전례)에도? 이중 변경 금지 — 팔 하나당 변경 하나.
5. **matched-baseline 프로토콜**: 같은 주·같은 리비전·같은 split의 large256-single 동시 재학습 + 동일 창 paired 비교(경로버그 §1 #11 재발 방지: ckpt 절대경로 실존 확인을 launch 스크립트에 내장), retry-supervisor 200ep 완주, run-tag provenance.

---

## 부록 — 콜리그(SigLIP 폴더)의 wrist 사용 (1문단)

콜리그는 wrist를 **조건 토큰 1개**로만 쓴다: 사실상 전 phase2 config에 `wrist_token: true` + `wrist_camera: eye_in_hand_rgb`이고, 인코딩은 **main과 같은 백본**(CLIP/SigLIP2/dual — 별도 wrist 인코더 없음, 변위 스트림 없음). 특징적으로 **augmentation A를 wrist에도 적용**(`wrist_augment: true, variants: 3`, 예 `SigLIP/configs/phase2_dualavg_rawconcat_bothaug_lang_zeropad.yaml:35-36`): base는 wrist blur/noise에 83→5–6.5%로 붕괴하나 both-aug로 87–89% 복구 + clean까지 +7pp(`SigLIP/result_report.md` §A). wrist의 정보 가치는 그들 진단에서도 최상위: grasp-상태 프로브 AUROC wrist 0.886–0.928 ≫ main 0.755–0.810(`SigLIP/report/S2_grasp_probe.md`), 정책 Jacobian 귀속 wrist 16.2–17.5%(`SigLIP/report/attribution.md`), mask-wrist 시 4% 붕괴, **wrist 토큰 스케일↑ 노브만이 유일하게 t4(서랍) 60→86 개선**(`SigLIP/report/post93_results.md:26`). 요약: 콜리그도 "wrist=고가치 조건 입력 + 강건화는 augmentation"이며, wrist를 추론/변위 스트림으로 승격한 시도는 없다.
