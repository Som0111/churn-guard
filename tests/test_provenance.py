"""The model report must name the exact data, code and library versions."""

from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from churnguard import config
from churnguard.train import build_pipeline, build_provenance, candidate_models

REQUIRED = {
    "model_version", "git_sha", "git_dirty", "trained_at", "dataset", "estimator",
    "estimator_hyperparameters", "calibration_method", "threshold_method", "splits",
    "dependencies", "python", "random_state",
}


def test_provenance_has_every_field():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            **{c: rng.uniform(1, 50, 60) for c in config.NUMERIC_FEATURES},
            **{c: rng.choice(["Yes", "No"], 60) for c in config.CATEGORICAL_FEATURES},
        }
    )
    y = pd.Series(rng.binomial(1, 0.3, 60))
    model = build_pipeline(candidate_models()["logistic_regression"]).fit(X, y)

    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    prov = build_provenance(model, "no_class_weight", {"train": y, "validation": y, "test": y}, when)

    assert REQUIRED <= prov.keys()
    assert prov["model_version"] == f"{prov['git_sha']}-20260102T030405Z"
    assert prov["dataset"]["sha256"] == config.DATA_SHA256 and prov["dataset"]["source_url"]
    assert prov["estimator"] == "LogisticRegression"
    assert set(prov["splits"]["train"]) == {"n", "churn_rate"}
    assert {"numpy", "pandas", "scikit-learn", "scipy", "joblib"} <= prov["dependencies"].keys()


# --------------------------------------------------------------------------- #
# Commit SHA in builds that have no .git
# --------------------------------------------------------------------------- #
import pytest

from churnguard import train


@pytest.fixture
def no_git(monkeypatch):
    monkeypatch.setattr(train, "_git", lambda *a: None)
    for name in ("GIT_SHA", "RENDER_GIT_COMMIT"):
        monkeypatch.delenv(name, raising=False)


def test_git_sha_env_is_used_and_shortened(no_git, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "0123456789abcdef0123456789abcdef01234567")
    assert train._commit_sha() == ("0123456", None)


def test_render_commit_is_used_when_git_sha_is_absent(no_git, monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "fedcba9876543210")
    assert train._commit_sha() == ("fedcba9", None)


@pytest.mark.parametrize("value", ["", "unknown", "  "])
def test_empty_or_unknown_values_are_treated_as_not_provided(no_git, monkeypatch, value):
    monkeypatch.setenv("GIT_SHA", value)
    assert train._commit_sha() == ("unknown", None)


def test_git_sha_beats_render_commit(no_git, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "aaaaaaa1111")
    monkeypatch.setenv("RENDER_GIT_COMMIT", "bbbbbbb2222")
    assert train._commit_sha()[0] == "aaaaaaa"


def test_git_is_used_when_the_environment_says_nothing(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    answers = {"rev-parse": "abc1234", "status": ""}  # keyed by subcommand: status carries pathspecs
    monkeypatch.setattr(train, "_git", lambda *a: answers[a[0]])
    assert train._commit_sha() == ("abc1234", False)


def test_model_version_carries_the_env_commit(no_git, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "deadbeefcafe")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            **{c: rng.uniform(1, 50, 60) for c in config.NUMERIC_FEATURES},
            **{c: rng.choice(["Yes", "No"], 60) for c in config.CATEGORICAL_FEATURES},
        }
    )
    y = pd.Series(rng.binomial(1, 0.3, 60))
    model = build_pipeline(candidate_models()["logistic_regression"]).fit(X, y)
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    prov = build_provenance(model, "no_class_weight", {"train": y, "validation": y, "test": y}, when)
    assert prov["model_version"] == "deadbee-20260102T030405Z"
    assert prov["git_dirty"] is None


# --------------------------------------------------------------------------- #
# git_dirty must describe the code, not the outputs of the training run
# --------------------------------------------------------------------------- #
def test_status_check_ignores_generated_output_paths(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    calls = []

    def fake_git(*args):
        calls.append(args)
        return "abc1234" if args[0] == "rev-parse" else ""

    monkeypatch.setattr(train, "_git", fake_git)
    assert train._commit_sha() == ("abc1234", False)
    status = next(c for c in calls if c[0] == "status")
    assert ":(exclude)reports" in status and ":(exclude)models" in status


def test_build_provenance_uses_the_commit_it_is_given(monkeypatch):
    def boom():
        raise AssertionError("must not look the commit up again after training wrote files")

    monkeypatch.setattr(train, "_commit_sha", boom)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            **{c: rng.uniform(1, 50, 60) for c in config.NUMERIC_FEATURES},
            **{c: rng.choice(["Yes", "No"], 60) for c in config.CATEGORICAL_FEATURES},
        }
    )
    y = pd.Series(rng.binomial(1, 0.3, 60))
    model = build_pipeline(candidate_models()["logistic_regression"]).fit(X, y)
    prov = build_provenance(
        model, "none", {"train": y, "validation": y, "test": y},
        datetime(2026, 1, 2, tzinfo=UTC), commit=("abc1234", False),
    )
    assert prov["git_sha"] == "abc1234" and prov["git_dirty"] is False


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_real_repo_generated_files_do_not_make_the_tree_dirty(tmp_path, monkeypatch):
    """A scratch repo: outputs under reports/ and models/ are ignored, code edits are not."""
    def run(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    run("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "metrics.json").write_text("{}")
    run("add", ".")
    run("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")

    for name in ("GIT_SHA", "RENDER_GIT_COMMIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert train._commit_sha()[1] is False

    (tmp_path / "reports" / "metrics.json").write_text('{"changed": true}')  # tracked output edited
    (tmp_path / "reports" / "new.png").write_text("png")                       # untracked output
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model_card.json").write_text("{}")
    assert train._commit_sha()[1] is False

    (tmp_path / "code.py").write_text("x = 2\n")  # a real code change
    assert train._commit_sha()[1] is True
