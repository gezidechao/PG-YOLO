"""Train, validate or benchmark PG-YOLO on the public dataset."""

import argparse
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "configs" / "training.yaml"
MODEL = HERE / "configs" / "ours_public_manual.yaml"
DEFAULT_DATA = ROOT / "datasets" / "new_OK2" / "data.yaml"
RUNS = ROOT / "runs" / "public"


def report_metrics(result):
    metrics = dict(result.results_dict)
    precision = metrics["metrics/precision(B)"]
    recall = metrics["metrics/recall(B)"]
    metrics["F1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "val", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--device", type=int, default=0, help="CUDA device index")
        if name != "benchmark":
            command.add_argument("--data", type=Path, default=DEFAULT_DATA)
        if name != "train":
            command.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    for name in ("data", "weights"):
        if hasattr(args, name):
            path = getattr(args, name).expanduser().resolve()
            if not path.is_file():
                parser.error(f"--{name} file not found: {path}")
            setattr(args, name, path)

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch
    import yaml
    from ultralytics import YOLO

    if args.command == "benchmark":
        from benchmark import benchmark

        torch.cuda.set_device(args.device)
        model = YOLO(str(args.weights)).model.float().cuda().eval()
        settings = {
            "imgsz": 1024,
            "warmup": 100,
            "iterations": 500,
            "repeats": 3,
            "fps_scope": "Unfused network forward only, GPU-resident input",
        }
        print(json.dumps(benchmark(model, settings), indent=2))
        return

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.command == "train":
        from bbox_instance_copy_paste import install

        install()
        torch.set_num_threads(4)
        torch.manual_seed(config["seed"])
        model = YOLO(str(MODEL), task="detect")
        report_metrics(model.train(
            **config, data=str(args.data), device=args.device,
            project=str(RUNS), name="pg_yolo_new_OK2",
        ))
    else:
        from ultralytics.models.yolo.detect import DetectionValidator

        options = {
            "model": str(args.weights), "task": "detect", "mode": "val",
            "data": str(args.data), "device": args.device, "split": "val",
            "imgsz": config["imgsz"], "batch": 2 * config["batch"],
            "rect": True, "half": True, "workers": 2 * config["workers"],
            "iou": config["iou"], "max_det": config["max_det"],
            "project": str(RUNS), "name": "val",
        }
        validator = DetectionValidator(args=options)
        validator(model=str(args.weights))
        report_metrics(validator.metrics)


if __name__ == "__main__":
    main()
