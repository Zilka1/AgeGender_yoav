"""Regression test for the notebooks' stage-level resume/restart-safety logic.

Both notebooks (notebooks/train_evaluate_{colab,kaggle}.ipynb) share an
identical "Training helpers" code cell defining experiment_paths,
train_one_experiment, calibrate_one_experiment, build_knn_one_experiment,
evaluate_one_experiment, and run_experiment_pipeline. Since notebook cells
aren't natively importable/unit-testable, this test extracts that cell's
*actual* source directly from the shipped .ipynb (not a reimplementation
that could silently drift from what's really there) and executes it in a
controlled namespace with a mocked run_command, to verify: with
FORCE_RERUN=False, a stage whose artifact already exists is skipped
(run_command not called for it), and a later stage whose artifact is
missing still runs -- i.e. an evaluation failure never causes training to
be redone once a checkpoint exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def _extract_training_helpers_source(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def run_experiment_pipeline" in source:
            return source
    raise AssertionError(f"No cell defining run_experiment_pipeline found in {notebook_path}")


@pytest.fixture(params=["train_evaluate_colab.ipynb", "train_evaluate_kaggle.ipynb"])
def training_helpers_namespace(request, tmp_path):
    """Exec the real notebook cell in a namespace with mocked I/O, returning
    (namespace, run_command_calls, run_dir) for assertions."""
    source = _extract_training_helpers_source(NOTEBOOKS_DIR / request.param)

    run_dir = tmp_path / "run"
    calls = []

    def fake_run_command(cmd, cwd=None, log_path=None, check=True, env=None):
        import torch

        calls.append(cmd)
        # Simulate scripts/train.py: creates the checkpoint the caller expects.
        # A real torch.save is needed since train_one_experiment loads it
        # back (torch.load(...)["config"]) to extract the resolved config.
        if cmd[1].endswith("train.py") or (len(cmd) > 1 and "train.py" in cmd[1]):
            experiment_name = cmd[cmd.index("--experiment-name") + 1]
            checkpoint_dir = run_dir / "experiments" / experiment_name / "seed_42" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"config": {}}, checkpoint_dir / f"{experiment_name}_best_balanced_score.pt")
        if "calibrate.py" in " ".join(str(c) for c in cmd):
            experiment_name = "fake_exp"
            calibration_dir = run_dir / "experiments" / experiment_name / "seed_42" / "calibration"
            calibration_dir.mkdir(parents=True, exist_ok=True)
            (calibration_dir / "conformal_calibration.json").write_text("{}", encoding="utf-8")
        if "evaluate.py" in " ".join(str(c) for c in cmd):
            experiment_name = "fake_exp"
            metrics_dir = run_dir / "experiments" / experiment_name / "seed_42" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / f"{experiment_name}_test_metrics.json").write_text('{"age_mae": 5.0}', encoding="utf-8")
        return 0, ""

    def fake_write_manifest(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")
        return Path(path)

    def fake_load_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def fake_validate_required_artifacts(paths):
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"missing: {missing}")
        return True

    def fake_flatten_overrides(obj, prefix=""):
        return []

    namespace = {
        "RUN_DIR": run_dir,
        "REPO_DIR": tmp_path,
        "FORCE_RERUN": False,
        "MAX_EPOCHS": 1,
        "EARLY_STOPPING_PATIENCE": 1,
        "experiments_cfg": {"fake_exp": {"overrides": {}}},
        "run_command": fake_run_command,
        "write_manifest": fake_write_manifest,
        "load_json": fake_load_json,
        "validate_required_artifacts": fake_validate_required_artifacts,
        "flatten_overrides": fake_flatten_overrides,
        "sys": __import__("sys"),
        "print": lambda *a, **k: None,  # silence the stage-plan/status prints
    }
    exec(compile(source, str(NOTEBOOKS_DIR / request.param), "exec"), namespace)
    return namespace, calls, run_dir


def test_first_run_executes_every_stage(training_helpers_namespace):
    namespace, calls, run_dir = training_helpers_namespace
    paths, metrics = namespace["run_experiment_pipeline"]("fake_exp", 42, include_knn=False)

    assert metrics == {"age_mae": 5.0}
    joined_calls = [" ".join(str(c) for c in call) for call in calls]
    assert any("train.py" in c for c in joined_calls)
    assert any("calibrate.py" in c for c in joined_calls)
    assert any("evaluate.py" in c for c in joined_calls)


def test_resume_skips_training_when_checkpoint_exists_but_reruns_missing_evaluation(training_helpers_namespace):
    """The core restart-safety requirement: pre-create a checkpoint (as if
    training previously succeeded) but no calibration/metrics (as if a
    later stage previously failed) -- re-running the pipeline must skip
    training (no train.py call) while still running calibration and
    evaluation."""
    namespace, calls, run_dir = training_helpers_namespace
    checkpoint_dir = run_dir / "experiments" / "fake_exp" / "seed_42" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "fake_exp_best_balanced_score.pt").write_bytes(b"already-trained")

    paths, metrics = namespace["run_experiment_pipeline"]("fake_exp", 42, include_knn=False)

    joined_calls = [" ".join(str(c) for c in call) for call in calls]
    assert not any("train.py" in c for c in joined_calls), "training must be skipped when checkpoint already exists"
    assert any("calibrate.py" in c for c in joined_calls), "calibration must still run since its artifact was missing"
    assert any("evaluate.py" in c for c in joined_calls), "evaluation must still run since its artifact was missing"
    assert metrics == {"age_mae": 5.0}


def test_resume_skips_every_stage_when_all_artifacts_already_exist(training_helpers_namespace):
    namespace, calls, run_dir = training_helpers_namespace
    base = run_dir / "experiments" / "fake_exp" / "seed_42"
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base / "checkpoints" / "fake_exp_best_balanced_score.pt").write_bytes(b"done")
    (base / "calibration").mkdir(parents=True, exist_ok=True)
    (base / "calibration" / "conformal_calibration.json").write_text("{}", encoding="utf-8")
    (base / "metrics").mkdir(parents=True, exist_ok=True)
    (base / "metrics" / "fake_exp_test_metrics.json").write_text('{"age_mae": 3.3}', encoding="utf-8")

    paths, metrics = namespace["run_experiment_pipeline"]("fake_exp", 42, include_knn=False)

    assert calls == [], "no stage should re-run when every artifact already exists"
    assert metrics == {"age_mae": 3.3}
