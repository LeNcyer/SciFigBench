"""Install each distribution independently and exercise public APIs without a model service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


def installed_smoke(source: Path) -> None:
    import scifigbench
    from scifigbench import SciFigBench, FullEvaluationRequired, evaluate, prepare_model_payload, load_qa
    from importlib.metadata import version
    assert scifigbench.__version__ == version("scifigbench")
    assert not Path(scifigbench.__file__).resolve().is_relative_to(source), "Imported the checkout instead of installed package"
    sys.path.insert(0, str(source / "tests"))
    from runner_support import HubStub, make_runner_dataset
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset, _ = make_runner_dataset(root / "source")
        hub = HubStub(dataset, root / "hub")
        hub.commit = scifigbench.runner.DEFAULT_DATASET_COMMIT
        calls = []
        def model(*, prompt, image):
            assert isinstance(prompt, str) and image.mode == "RGB"
            calls.append(1)
            return "invalid-output"
        previous = Path.cwd()
        try:
            os.chdir(root)
            with patch.dict(sys.modules, {"huggingface_hub": hub.module()}), patch("pathlib.Path.home", return_value=root / "home"):
                result = SciFigBench(model=model)
                assert len(calls) == 25 and result.evaluation_mode == "quick"
                data = result.report["dataset"]["data_root"]
                assert result.report["evaluation"] == evaluate(result.predictions_path, data_root=data, allow_partial=True)
                payload = prepare_model_payload(load_qa(data)[0], data_root=data)
                assert set(payload) == {"prompt", "image"}
                payload["image"].close()
                calls.clear()
                try:
                    SciFigBench(model=model, max_samples_per_task=None)
                except FullEvaluationRequired as exc:
                    assert "30 QA" in str(exc)
                else:
                    raise AssertionError("Missing full-run confirmation did not fail")
                assert not calls
                result = SciFigBench(model=model, local_files_only=True)
                assert len(calls) == 25 and not result.is_full_benchmark
        finally:
            os.chdir(previous)
    subprocess.run([sys.executable, "-m", "scifigbench", "--help"], check=True, stdout=subprocess.DEVNULL)
    cli = Path(sys.executable).parent / ("scifigbench.exe" if os.name == "nt" else "scifigbench")
    subprocess.run([str(cli), "--help"], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({"python": sys.version.split()[0], "installed": str(Path(scifigbench.__file__)),
                      "quick_calls": 25, "cli": "pass", "api": "pass", "offline": "pass"}))


def install_smoke(dist: Path, source: Path, wheelhouse: Path | None) -> None:
    artifacts = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
    assert len(artifacts) == 2, "Expected wheel and sdist"
    for artifact in artifacts:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = root / "venv"
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            env = dict(os.environ)
            for name in ("PYTHONPATH", "SCIFIGBENCH_DATA_DIR", "SCIFIGBENCH_REPO_ID", "SCIFIGBENCH_REVISION"):
                env.pop(name, None)
            if wheelhouse is not None:
                env["PIP_NO_INDEX"] = "1"
                env["PIP_FIND_LINKS"] = str(wheelhouse.resolve())
            subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                            "--no-input", str(artifact.resolve())], check=True, env=env, cwd=root)
            subprocess.run([str(python), str(Path(__file__).resolve()), "--installed", "--source", str(source)],
                           check=True, env=env, cwd=root)
            print(f"Installed distribution passed: {artifact.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    if args.installed:
        installed_smoke(args.source.resolve())
    else:
        install_smoke(args.dist.resolve(), args.source.resolve(), args.wheelhouse)
