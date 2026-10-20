"""The console command and python -m entry point share the public Python API."""
import argparse
import json
from pathlib import Path
from .data import DataError
from .tasks import TASKS
from .download import download_data
from .evaluation import evaluate
from ._version import __version__

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="scifigbench", description="Download SciFigBench data and score model predictions.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    dl = commands.add_parser("download", help="Explicitly download or reuse QA, Images and optional Annotations")
    dl.add_argument("--repo-id")
    dl.add_argument("--revision")
    dl.add_argument("--data-root", type=Path)
    dl.add_argument("--cache-dir", type=Path)
    modes = dl.add_mutually_exclusive_group()
    modes.add_argument("--annotations", action="store_true")
    modes.add_argument("--qa-only", action="store_true")
    dl.add_argument("--local-files-only", action="store_true")
    ev = commands.add_parser("evaluate", help="Evaluate selected tasks; full prediction coverage is required by default")
    ev.add_argument("--predictions", type=Path, required=True)
    ev.add_argument("--task", choices=["all", *TASKS], default="all")
    ev.add_argument("--data-root", type=Path)
    ev.add_argument("--allow-partial", action="store_true")
    ev.add_argument("--prediction-field", default="response")
    ev.add_argument("--include-details", action="store_true")
    ev.add_argument("--no-latex-normalize", action="store_true")
    ev.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            root = download_data(repo_id=args.repo_id, revision=args.revision, local_dir=args.data_root,
                                 cache_dir=args.cache_dir, annotations=args.annotations, qa_only=args.qa_only,
                                 local_files_only=args.local_files_only)
            result = {"data_root": str(root)}
        else:
            result = evaluate(args.predictions, data_root=args.data_root, task=args.task, allow_partial=args.allow_partial,
                              prediction_field=args.prediction_field, include_details=args.include_details,
                              no_latex_normalize=args.no_latex_normalize)
        text = json.dumps(result, ensure_ascii=False, indent=2)+"\n"
        if args.command == "evaluate" and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
    except (DataError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 0
