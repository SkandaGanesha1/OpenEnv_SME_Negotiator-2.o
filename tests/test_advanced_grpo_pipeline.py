"""Regression tests for the advanced GRPO orchestration and notebook."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.demo import inspect_policy_reports, save_run_manifest
from rl.train_grpo_advanced import build_advanced_run_plan
from rl.train_grpo_trl import resolve_training_precision_kwargs


def _workspace_tmp_dir(name: str) -> Path:
    path = PROJECT_ROOT / ".test_tmp" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_notebook_source(name: str) -> str:
    notebook_path = PROJECT_ROOT / "notebooks" / name
    payload = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))


def test_resolve_training_precision_kwargs_prefers_fp16_on_non_bf16_cuda(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    fake_torch.cuda = _Cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_training_precision_kwargs() == {"bf16": False, "fp16": True}


def test_inspect_policy_reports_flags_malformed_and_suspicious_completions() -> None:
    reports = [
        {
            "completion_records": [
                {
                    "step": 1,
                    "valid_json": False,
                    "raw_text": "<think>globals()['hack'] = 1</think>",
                    "action_type": "proposal",
                    "tool_name": "QUERY_TREDS",
                },
                {
                    "step": 2,
                    "valid_json": True,
                    "raw_text": '{"action_type":"tool","tool_name":"QUERY_TREDS","tool_args":{}}',
                    "action_type": "tool",
                    "tool_name": "QUERY_TREDS",
                },
            ]
        }
    ]

    inspection = inspect_policy_reports(reports, policy_label="tiny_trl")

    assert inspection["invalid_parse_fraction"] == 0.5
    assert inspection["unsupported_action_count"] == 1
    assert inspection["duplicate_tool_abuse_count"] == 1
    assert inspection["suspicious_pattern_count"] == 1
    assert inspection["flagged"] is True


def test_advanced_plan_and_manifest_include_expected_phases() -> None:
    tmp_path = _workspace_tmp_dir("advanced_manifest")
    plan = build_advanced_run_plan(profile="submission", output_dir=str(tmp_path))
    manifest_path = save_run_manifest({"plan": plan}, output_dir=str(tmp_path))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    assert Path(manifest_path).exists()
    assert plan["phases"] == [
        "tiny_trl_run",
        "tiny_unsloth_run",
        "tiny_policy_comparison",
        "hacking_inspection",
        "curriculum_decision",
        "main_trl_run",
        "before_after_evaluation",
        "manifest_export",
    ]
    assert manifest["plan"]["paths"]["run_manifest"].endswith("run_manifest.json")
    assert manifest["plan"]["task"]["task_name"] == "liquidity-correlation-hard"


def test_liquidity_notebook_uses_simple_package_driven_training_flow() -> None:
    source = _load_notebook_source("grpo_sme_liquidity.ipynb")

    assert 'RUN_PROFILE = "tiny"' in source
    assert "from rl.train_grpo_liquidity import (" in source
    assert "make_training_args" in source
    assert "build_canonical_training_args" in source
    assert "build_run_plan" in source
    assert "build_training_session" in source
    assert "smoke_test_environment" in source
    assert "run_training" in source
    assert "plot_rewards" in source
    assert "Smoke test passed. Environment is ready for training." in source
    assert 'outputs/grpo_sme_liquidity_simple' in source
    assert "Checkpoint path:" in source

