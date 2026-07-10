"""LIBERO-Para paraphrases for the 10 libero_goal tasks (experiment 1b).

Object-lexical vs action-lexical paraphrase robustness. Strings are frozen in
configs/libero_para_goal.json (source: GitHub cau-hai-lab/LIBERO-Para @ master,
metrics/libero_para_metadata.csv, MIT license). Bundled under configs/ (not
data/, which is git-ignored) so it is tracked and syncs to the remote box.

Axis definition (both are LEXICAL-level edits; only the target differs):
  object -> high=obj, mid=lexical  (rewords the OBJECT noun phrase,
            e.g. "cabinet" -> "storage cabinet" / "filing cabinet")
  action -> high=act, mid=lexical  (rewords/modifies the ACTION verb,
            e.g. "open" -> "gently open" / "carefully open")

IMPORTANT — index alignment: LIBERO-Para's eval{i} index is NOT our libero_goal
task_id. This dict is keyed by OUR task_id (benchmark.get_benchmark_dict()
['libero_goal']().get_task(i).language), mapped from LIBERO-Para by exact
base-string match. Per-task lp_eval is recorded in the JSON _meta / tasks.

usage:
  from eval_libero.libero_para import LIBERO_PARA, BASE
  LIBERO_PARA[task_id]["object"]  # list[str]
  LIBERO_PARA[task_id]["action"]  # list[str]
"""
import json
from pathlib import Path

_WS = Path(__file__).resolve().parents[2]
_JSON = _WS / "configs" / "libero_para_goal.json"

_data = json.loads(_JSON.read_text())
META = _data["_meta"]

# LIBERO_PARA[task_id] = {"object": [...], "action": [...]}
LIBERO_PARA = {
    int(tid): {"object": t["object"], "action": t["action"]}
    for tid, t in _data["tasks"].items()
}
# task_id -> original (correct) libero_goal instruction
BASE = {int(tid): t["base"] for tid, t in _data["tasks"].items()}
# task_id -> LIBERO-Para eval index (for traceability)
LP_EVAL = {int(tid): t["lp_eval"] for tid, t in _data["tasks"].items()}

AXES = ("object", "action")


def counts():
    """{task_id: {"object": n, "action": n}} — for load-tests / logging."""
    return {tid: {ax: len(v[ax]) for ax in AXES} for tid, v in LIBERO_PARA.items()}
