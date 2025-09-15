import yaml
import os
from typing import Any, Dict, Optional

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_config_profile() -> Dict[str, Any]:
    """Load the full configuration profile (levels, thresholds, policy, format_defaults_by_tier)."""
    cfg_path = "config/config.yaml"
    data = load_yaml(cfg_path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config at {cfg_path}: expected mapping at root")
    return data

def load_questions():
    data = load_yaml("questions/bank.yaml")
    return [item for item in data if item["kind"] == "question"]

def load_scenarios():
    data = load_yaml("questions/bank.yaml")
    return [item for item in data if item["kind"] == "scenario"]