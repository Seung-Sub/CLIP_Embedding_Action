import os
from pathlib import Path

import yaml

_CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _expand(node):
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, str) and node.startswith("~"):
        return os.path.expanduser(node)
    return node


def load_config(path=None):
    with open(path or _CFG_PATH) as f:
        return _expand(yaml.safe_load(f))
