"""Stage 5 dry-run and helper tests for RL training scripts."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.curriculum import CurriculumManager
from rl.opponents import OpponentPolicyManager
from rl.episode_logging import EpisodeSummary
from rl.train_grpo_trl import (
    EpisodeSummaryBuffer,
    build_arg_parser,
    build_environment_factory,
    build_grpo_config_kwargs,
    build_metrics_callback,
    build_snapshot_callback,
    build_training_rows,
    configure_tokenizer,
    main as trl_main,
    make_reward_function,
)
from rl.train_grpo_unsloth import main as unsloth_main


class _DummyTokenizer:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "<eos>"


class _DummyEnv:
    current_persona = None

    def compute_final_reward(self) -> float:
        return 0.75

    def build_episode_log(self) -> str:
        return "episode-log"

    def summarize_episode(self) -> EpisodeSummary:
        return EpisodeSummary(
            episode_completed=True,
            base_rl_reward=0.7,
            tool_bonus_total=0.05,
            env_reward_total=0.75,
            success_no_default_positive_npv=True,
            average_final_payment_days=30.0,
            tool_usage_count=2,
            resolved_deal_count=2,
            defaulted_sme_count=0,
        )


class _DummyModelSaver:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_text("model", encoding="utf-8")


class _DummyTokenizerSaver:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("tokenizer", encoding="utf-8")


def _workspace_tmp_dir(name: str) -> Path:
    path = PROJECT_ROOT / ".test_tmp" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_dry_run_works_for_trl_and_unsloth(capsys) -> None:
    assert trl_main(["--dry-run", "--num-samples", "2"]) == 0
    captured = capsys.readouterr()
    assert "liquidity-correlation-hard" in captured.out

    assert unsloth_main(["--dry-run", "--num-samples", "2"]) == 0
    captured = capsys.readouterr()
    assert "liquidity-correlation-hard" in captured.out


def test_dry_run_supports_stage6_flags(capsys) -> None:
    assert (
        trl_main(
            [
                "--dry-run",
                "--num-samples",
                "2",
                "--enable-self-play",
                "--enable-rubrics",
                "--persona-mode",
                "fixed",
                "--build-preference-dataset",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"self_play"' in captured.out
    assert '"rubrics"' in captured.out

    assert (
        unsloth_main(
            [
                "--dry-run",
                "--num-samples",
                "2",
                "--enable-self-play",
                "--enable-rubrics",
                "--persona-mode",
                "fixed",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"self_play"' in captured.out
    assert '"curriculum"' in captured.out


def test_dataset_builder_emits_expected_columns_and_seed_range() -> None:
    rows = build_training_rows(num_samples=3, seed_base=1000)
    assert len(rows) == 3
    assert rows[0]["seed"] == 1000
    assert rows[2]["seed"] == 1002
    assert set(rows[0]) == {"prompt", "task_name", "difficulty", "seed", "total_periods"}


def test_configure_tokenizer_enforces_left_padding_and_pad_token_fallback() -> None:
    tokenizer = configure_tokenizer(_DummyTokenizer())
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"


def test_reward_function_reads_environments_and_returns_one_scalar_per_env() -> None:
    reward_func = make_reward_function(
        rubric_scorer=lambda episode_log: {"rubric": 0.5},
        rubric_weight=0.2,
    )
    rewards = reward_func([_DummyEnv(), _DummyEnv()])
    assert rewards == [0.85, 0.85]


def test_snapshot_callback_registers_and_prunes_opponent_zoo() -> None:
    tmp_path = _workspace_tmp_dir("snapshot_callback")
    manager = OpponentPolicyManager(snapshots_dir=tmp_path / "snapshots", zoo_size=2)
    callback = build_snapshot_callback(
        object,
        opponent_manager=manager,
        interval=100,
        output_dir=str(tmp_path),
    )

    for step in (100, 200, 300):
        callback.on_step_end(
            None,
            SimpleNamespace(global_step=step),
            SimpleNamespace(),
            model=_DummyModelSaver(),
            processing_class=_DummyTokenizerSaver(),
        )

    assert manager.snapshot_ids() == [
        "sme_policy_step_000200",
        "sme_policy_step_000300",
    ]


def test_environment_factory_picks_up_curriculum_config_and_opponents() -> None:
    tmp_path = _workspace_tmp_dir("environment_factory")
    curriculum = CurriculumManager(window_size=1)
    curriculum.record_episode(0.9, False)
    assert curriculum.maybe_advance_level() is True

    opponent_manager = OpponentPolicyManager(snapshots_dir=tmp_path / "snapshots", zoo_size=2)
    args = Namespace(
        task_name="liquidity-correlation-hard",
        difficulty="hard",
        total_periods=3,
        seed_base=1000,
        persona_mode="off",
        persona_name=None,
    )
    env_factory = build_environment_factory(
        args,
        curriculum=curriculum,
        opponent_manager=opponent_manager,
    )

    wrapper = env_factory()
    wrapper.reset(
        prompt=[{"role": "user", "content": "Train me"}],
        task_name="liquidity-correlation-hard",
        difficulty="hard",
        seed=1000,
        total_periods=99,
    )

    assert wrapper.total_periods == curriculum.current_config().total_periods
    assert wrapper.buyer_variance == curriculum.current_config().buyer_variance
    assert wrapper.financier_variance == curriculum.current_config().financier_variance
    assert wrapper.curriculum_level == curriculum.current_level()
    assert wrapper.opponent_manager is opponent_manager


def test_build_grpo_config_kwargs_uses_training_log_backend_env(monkeypatch) -> None:
    args = build_arg_parser().parse_args([])

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "tensorboard")
    assert build_grpo_config_kwargs(args)["report_to"] == "tensorboard"

    monkeypatch.setenv("TRAINING_LOG_BACKEND", "unsupported")
    assert build_grpo_config_kwargs(args)["report_to"] == "none"


def test_metrics_callback_saves_reward_curve_without_matplotlib_dependency(monkeypatch) -> None:
    tmp_path = _workspace_tmp_dir("metrics_callback")
    summary_buffer = EpisodeSummaryBuffer()
    callback = build_metrics_callback(
        summary_buffer,
        object,
        curriculum=None,
        build_preference_dataset=False,
        scorer=None,
        output_dir=str(tmp_path),
    )

    def _fake_save(reward_curve, success_curve, *, output_dir):
        path = Path(output_dir) / "reward_curve.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr("rl.train_grpo_trl._save_reward_curve_plot", _fake_save)

    control = SimpleNamespace()
    args = SimpleNamespace(output_dir=str(tmp_path))
    for index, reward in enumerate((0.2, 0.5), start=1):
        summary_buffer.append(
            EpisodeSummary(
                episode_completed=True,
                base_rl_reward=reward,
                verifiable_reward=reward,
                total_reward=reward,
                tool_bonus_total=0.01,
                env_reward_total=reward,
                success_no_default_positive_npv=True,
                average_final_payment_days=40.0,
                tool_usage_count=1,
                resolved_deal_count=2,
                defaulted_sme_count=0,
            )
        )
        callback.on_log(args, SimpleNamespace(global_step=index), control, logs={})

    callback.on_train_end(args, SimpleNamespace(global_step=2), control)

    assert (tmp_path / "reward_curve.png").exists()


def test_metrics_callback_plot_failures_do_not_raise(monkeypatch) -> None:
    tmp_path = _workspace_tmp_dir("metrics_callback_failure")
    summary_buffer = EpisodeSummaryBuffer()
    callback = build_metrics_callback(
        summary_buffer,
        object,
        curriculum=None,
        build_preference_dataset=False,
        scorer=None,
        output_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        "rl.train_grpo_trl._save_reward_curve_plot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )

    control = SimpleNamespace()
    args = SimpleNamespace(output_dir=str(tmp_path))
    for index, reward in enumerate((0.2, 0.4), start=1):
        summary_buffer.append(
            EpisodeSummary(
                episode_completed=True,
                base_rl_reward=reward,
                verifiable_reward=reward,
                total_reward=reward,
                tool_bonus_total=0.0,
                env_reward_total=reward,
                success_no_default_positive_npv=True,
                average_final_payment_days=45.0,
                tool_usage_count=0,
                resolved_deal_count=1,
                defaulted_sme_count=0,
            )
        )
        callback.on_log(args, SimpleNamespace(global_step=index), control, logs={})

    callback.on_train_end(args, SimpleNamespace(global_step=2), control)
