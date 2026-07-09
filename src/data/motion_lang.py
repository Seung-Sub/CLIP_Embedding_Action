"""C8 — 청크 수준 모션 문장 생성기 (ARM-DA/HY의 정렬 타깃).

카테고리 = 지배축(x/y/z/roll/pitch/yaw) × 방향(±) × 크기(2-bin) + 그리퍼 단독(open/close).
어휘는 data/motion_lang.json에 고정(최초 1회 생성 후 불변) — 재현성·감사 가능성 확보.

설계 근거 (outputs/report/c8_chunk_axis_stats.json, 2026-07-06):
  - 크기 bin 경계 9.65 = 지배축 |누적변위| 중앙값 (bin 균형 50/50, 임계 자의성 제거)
  - 지배축 분포 z 56% / x 24% / y 19% / 회전 <1.3% → 회전 문장은 완비성 위해 포함하되
    학습 중 등장 희소함을 명기 (zero-shot hold-out은 pos 축 위주로 구성)
  - 그리퍼 이벤트율 25% → 이동 문장 + 단일 표준 접미사(" and close/open the gripper")로
    조합 (조합 폭발 방지, 고유 문장 수 = base×3 ≤ 510)
  - 방향 단어 규약(robosuite OSC 월드 프레임): +x 전진 / +y 좌 / +z 상승.
    내부 일관 라벨로 사용 — 절제 비교엔 프레임 부호의 절대 의미 불필요.

금지 사항(계획 지시): task.language 정렬 변형 제작 금지 — 청크 모션 문장 전용.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
VOCAB_PATH = WS / "data" / "motion_lang.json"
MAG_BOUNDARY = 9.65          # c8_chunk_axis_stats.json 중앙값 (고정)

AXES = ["x", "y", "z", "roll", "pitch", "yaw"]
DIR_WORDS = {
    ("x", 1): "forward", ("x", -1): "backward",
    ("y", 1): "to the left", ("y", -1): "to the right",
    ("z", 1): "upward", ("z", -1): "downward",
}
ROT_WORDS = {
    ("roll", 1): "roll the wrist to the left", ("roll", -1): "roll the wrist to the right",
    ("pitch", 1): "tilt the wrist up", ("pitch", -1): "tilt the wrist down",
    ("yaw", 1): "turn the wrist counterclockwise", ("yaw", -1): "turn the wrist clockwise",
}
GRIP_SUFFIX = {1: " and close the gripper", 2: " and open the gripper"}


def _build_vocab():
    """템플릿(동사×부사 골격 ~50종) + 저자 패러프레이즈 → 카테고리별 train/holdout 문장."""
    verbs = ["move the gripper", "move the arm", "shift the end effector",
             "push the hand", "slide the gripper", "bring the arm"]
    small = ["slightly", "a little", "a bit"]
    large = ["a lot", "far", "a long distance"]
    # hold-out: 템플릿 골격과 어휘가 다른 수작업 문장 (일반화 검증용)
    manual_holdout = {
        "x+|L": "advance the hand a great deal toward the front",
        "x+|S": "nudge the hand forward just a touch",
        "x-|L": "pull the arm way back",
        "x-|S": "ease the gripper back a tiny bit",
        "y+|L": "swing the arm out wide to the left side",
        "y+|S": "scoot the hand left a smidge",
        "y-|L": "sweep the gripper right across the table",
        "y-|S": "inch the arm over to the right",
        "z+|L": "raise the hand high up",
        "z+|S": "lift the gripper up a touch",
        "z-|L": "drop the arm all the way down",
        "z-|S": "lower the hand a hair closer to the table",
        "roll+|L": "rotate the wrist strongly leftward",
        "pitch-|L": "angle the wrist sharply downward",
        "yaw+|L": "spin the wrist a good amount counterclockwise",
        "grip+": "clamp the fingers shut",
        "grip-": "release the grasp",
        "z-|S#2": "descend gently toward the surface",
        "x+|L#2": "drive the gripper firmly ahead",
        "y+|S#2": "drift the hand slightly leftward",
    }
    cats = {}
    for ax in AXES:
        for sgn in (1, -1):
            for mag, advs in (("S", small), ("L", large)):
                key = f"{ax}{'+' if sgn > 0 else '-'}|{mag}"
                sents = []
                if ax in ("x", "y", "z"):
                    d = DIR_WORDS[(ax, sgn)]
                    for i, v in enumerate(verbs):
                        sents.append(f"{v} {advs[i % len(advs)]} {d}")
                else:
                    base = ROT_WORDS[(ax, sgn)]
                    for adv in advs:
                        sents.append(f"{base} {adv}")
                cats[key] = {"train": sents, "holdout": []}
    cats["grip+"] = {"train": ["close the gripper", "grasp with the gripper",
                               "shut the gripper fingers"], "holdout": []}
    cats["grip-"] = {"train": ["open the gripper", "let go with the gripper",
                               "spread the gripper fingers"], "holdout": []}
    for k, s in manual_holdout.items():
        cats[k.split("#")[0]]["holdout"].append(s)
    n_train = sum(len(c["train"]) for c in cats.values())
    n_hold = sum(len(c["holdout"]) for c in cats.values())
    return {"mag_boundary": MAG_BOUNDARY, "dir_convention": dict(
                (f"{a}{'+' if s>0 else '-'}", w) for (a, s), w in
                list(DIR_WORDS.items()) + list(ROT_WORDS.items())),
            "grip_suffix": {str(k): v for k, v in GRIP_SUFFIX.items()},
            "counts": {"train": n_train, "holdout": n_hold},
            "categories": cats}


def _build_vocab_v2():
    """F2.5 증강판 — prior 학습 전용 (phase1 팔은 v1 고정, hold-out 불가침).

    v1 대비: 방향 동의어·방향 전용 동사·부사 확대 → 카테고리당 문장 다양성 ~4배.
    hold-out 20종과의 완전 일치 문장은 생성 후 제거로 보장.
    """
    v1 = _build_vocab()
    holdout_all = {s for c in v1["categories"].values() for s in c["holdout"]}
    subjects = ["the gripper", "the arm", "the hand", "the end effector"]
    small = ["slightly", "a little", "a bit", "a touch", "just a bit"]
    large = ["a lot", "far", "way out", "considerably", "all the way"]
    dir_words = {("x", 1): ["forward", "ahead", "to the front"],
                 ("x", -1): ["backward", "back", "to the rear"],
                 ("y", 1): ["to the left", "leftward", "left"],
                 ("y", -1): ["to the right", "rightward", "right"],
                 ("z", 1): ["upward", "up", "higher"],
                 ("z", -1): ["downward", "down", "lower"]}
    dir_verbs = {("x", 1): ["advance", "push"], ("x", -1): ["retract", "pull"],
                 ("y", 1): ["swing", "shift"], ("y", -1): ["swing", "shift"],
                 ("z", 1): ["raise", "lift"], ("z", -1): ["lower", "drop"]}
    base_verbs = ["move", "shift", "slide", "bring", "take"]
    cats = {}
    for ax in AXES:
        for sgn in (1, -1):
            for mag, advs in (("S", small), ("L", large)):
                key = f"{ax}{'+' if sgn > 0 else '-'}|{mag}"
                sents = list(v1["categories"][key]["train"])
                if ax in ("x", "y", "z"):
                    for i, v in enumerate(base_verbs + dir_verbs[(ax, sgn)]):
                        for j, d in enumerate(dir_words[(ax, sgn)]):
                            adv = advs[(i + j) % len(advs)]
                            subj = subjects[(i * 3 + j) % len(subjects)]
                            sents.append(f"{v} {subj} {adv} {d}")
                else:
                    base = ROT_WORDS[(ax, sgn)]
                    sents += [f"{base} {a}" for a in advs]
                sents = [x for x in dict.fromkeys(sents) if x not in holdout_all]
                cats[key] = {"train": sents,
                             "holdout": v1["categories"][key]["holdout"]}
    for gk, extra in (("grip+", ["close the fingers", "squeeze the gripper shut",
                                 "grip the object", "tighten the gripper"]),
                      ("grip-", ["open the fingers", "let the object go",
                                 "release the gripper", "widen the gripper"])):
        sents = v1["categories"][gk]["train"] + extra
        cats[gk] = {"train": [x for x in sents if x not in holdout_all],
                    "holdout": v1["categories"][gk]["holdout"]}
    n_train = sum(len(c["train"]) for c in cats.values())
    return {**{k: v for k, v in v1.items() if k != "categories"},
            "version": "v2-f25", "counts": {"train": n_train,
            "holdout": sum(len(c["holdout"]) for c in cats.values())},
            "grip_only_rule": "grip 이벤트 AND 지배축 |누적| < 경계*0.3 → grip 단독 문장",
            "categories": cats}


def _build_vocab_v3():
    """H2-fair §5: dual-score 선별 — CLIP·SigLIP2 양쪽에서 카테고리 마진>0 문장만
    유지 (v2 풀 필터, 카테고리당 최소 3문장 = min-margin 상위 폴백). hold-out 불가침."""
    import numpy as np
    import torch
    from core.anchor import ClipAnchor, Siglip2Anchor
    v2 = load_vocab("v2")
    cats = v2["categories"]
    names, sents = [], []
    for c, d in cats.items():
        for x in d["train"]:
            names.append(c); sents.append(x)
    margins = {}
    for aname, cls in (("clip", ClipAnchor), ("siglip2", Siglip2Anchor)):
        a = cls(normalize=False)
        E = np.concatenate([a.encode_texts(sents[i:i+64])["embeds"]
                            for i in range(0, len(sents), 64)])
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        keys = list(cats)
        cent = np.stack([En[[i for i, n in enumerate(names) if n == c]].mean(0)
                         for c in keys])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
        sim = En @ cent.T
        own = np.array([sim[i, keys.index(names[i])] for i in range(len(sents))])
        oth = np.array([np.delete(sim[i], keys.index(names[i])).max()
                        for i in range(len(sents))])
        margins[aname] = own - oth
        del a
        torch.cuda.empty_cache()
    keep = (margins["clip"] > 0) & (margins["siglip2"] > 0)
    score = np.minimum(margins["clip"], margins["siglip2"])
    out = {}
    for c in cats:
        idx = [i for i, n in enumerate(names) if n == c]
        kept = [sents[i] for i in idx if keep[i]]
        if len(kept) < 3:
            kept = [sents[i] for i in sorted(idx, key=lambda i: -score[i])[:3]]
        out[c] = {"train": kept, "holdout": cats[c]["holdout"]}
    return {**{k: v for k, v in v2.items() if k != "categories"},
            "version": "v3-dual",
            "selection": "margin>0 in BOTH clip & siglip2 (min-margin top-3 폴백)",
            "counts": {"train": sum(len(d["train"]) for d in out.values()),
                       "holdout": v2["counts"]["holdout"]},
            "categories": out}


def load_vocab(version="v1"):
    path = {"v1": VOCAB_PATH,
            "v2": VOCAB_PATH.with_name("motion_lang_v2.json"),
            "v3": VOCAB_PATH.with_name("motion_lang_v3.json")}[version]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        builder = {"v1": _build_vocab, "v2": _build_vocab_v2,
                   "v3": _build_vocab_v3}[version]
        path.write_text(json.dumps(builder(), indent=1))
        print(f"{path.name} 생성(고정)")
    return json.loads(path.read_text())


def chunk_category(chunk_raw):
    """비정규화 청크 (T, 7) → (카테고리 키, grip 코드 0/1/2).

    지배축 = |축별 누적변위| argmax. 그리퍼: 부호 변화 이벤트 (마지막 방향 기준).
    """
    cum = chunk_raw[:, :6].sum(0)
    ax = int(np.abs(cum).argmax())
    sgn = 1 if cum[ax] > 0 else -1
    mag = "L" if np.abs(cum[ax]) >= MAG_BOUNDARY else "S"
    g = chunk_raw[:, 6]
    grip = 0
    sign_change = np.diff(np.sign(g)) != 0
    if sign_change.any():
        grip = 1 if g[-1] > 0 else 2          # 관례: +1=close, -1=open
    return f"{AXES[ax]}{'+' if sgn > 0 else '-'}|{mag}", grip


class MotionSentences:
    """청크 → 고정 문장 할당 + 전체 고유 문장 임베딩 테이블."""

    def __init__(self, version="v1"):
        self.version = version
        self.vocab = load_vocab(version)
        self.sentences = []                   # 고유 문장 (train 조합 전체)
        self.index = {}                       # (cat, variant, grip) -> sent_id
        for cat, c in self.vocab["categories"].items():
            for vi, s in enumerate(c["train"]):
                grips = (0, 1, 2) if not cat.startswith("grip") else (0,)
                for g in grips:
                    self.index[(cat, vi, g)] = len(self.sentences)
                    self.sentences.append(s + GRIP_SUFFIX.get(g, ""))
        self.holdout = [(cat, s) for cat, c in self.vocab["categories"].items()
                        for s in c["holdout"]]

    def assign(self, chunks_raw):
        """(N, T, 7) → sent_ids (N,). 변형 선택 = 결정론 해시(에폭 불변, 캐시 친화)."""
        ids = np.empty(len(chunks_raw), dtype=np.int64)
        for i, ch in enumerate(chunks_raw):
            cat, grip = chunk_category(ch)
            if (self.version != "v1" and grip != 0
                    and np.abs(ch[:, :6].sum(0)).max() < MAG_BOUNDARY * 0.3):
                cat = "grip+" if grip == 1 else "grip-"   # v2: grip 단독 할당 (버그 수정)
            n_var = len(self.vocab["categories"][cat]["train"])
            if cat.startswith("grip"):
                grip = 0
            h = int(hashlib.md5(ch.tobytes()).hexdigest()[:8], 16)
            ids[i] = self.index[(cat, h % n_var, grip)]
        return ids

    def embed_all(self, anchor):
        """전체 고유 문장 임베딩 (N_sent, dim_text) — 앵커 텍스트 인코더."""
        out = []
        for i in range(0, len(self.sentences), 64):
            out.append(anchor.encode_texts(self.sentences[i:i + 64])["embeds"])
        return np.concatenate(out).astype(np.float32)

    def category_of_sent(self, sent_id):
        for (cat, vi, g), sid in self.index.items():
            if sid == sent_id:
                return cat, g
        raise KeyError(sent_id)
