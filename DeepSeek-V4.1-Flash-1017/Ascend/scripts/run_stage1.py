#!/usr/bin/env python3
"""Prepare and quantize a fresh stage1 workspace; never delete prior outputs."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOL_COMMIT = "c61748f295dc8778a978fbf97ee790609f9ed2ea"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="Downloaded official 1017 checkpoint")
    ap.add_argument("--tool", required=True, type=Path, help="Patched msmodelslim checkout")
    ap.add_argument("--work", required=True, type=Path, help="New, nonexistent workspace")
    ap.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--layers", type=int, default=40, choices=range(1, 41))
    ap.add_argument("--dry-run", action="store_true", help="Print commands without writing or using NPU")
    args = ap.parse_args()
    model, tool, work = (p.resolve() for p in (args.model, args.tool, args.work))
    if args.work.exists() or args.work.is_symlink():
        ap.error("--work already exists; choose a new directory")
    if min(args.devices) < 0 or len(set(args.devices)) != len(args.devices):
        ap.error("device IDs must be unique nonnegative integers")
    prepared, output = work / "input-stage1", work / "stage1"
    recipe = work / "deepseek_v4_1_flash_w4a8.yaml"
    cli = Path(sys.executable).with_name("msmodelslim")
    commands = [
        [sys.executable, str(ROOT / "scripts/make_full_model.py"),
         "--src", str(model), "--dst", str(prepared), "--layers", str(args.layers),
         "--mtp", "0", "--engram", "off", "--vision", "drop"],
        [str(cli), "quant", "--model_path", str(prepared),
         "--save_path", str(output), "--config", str(recipe), "--device", "npu",
         "--device_id", *map(str, args.devices), "--model_type", "deepseek_v41",
         "--trust_remote_code", "true", "--log_level", "info"],
        [sys.executable, str(ROOT / "scripts/prepare_runtime_ckpt.py"), "--dir", str(output)],
    ]
    if args.dry_run:
        print("Dry run: no files written, no NPU operations. Runtime recipe uses an absolute calibration path.")
        for command in commands:
            print(shlex.join(command))
        return
    if not cli.is_file():
        ap.error(f"missing console entry {cli}; install msmodelslim in this Python environment")
    actual = subprocess.check_output(["git", "-C", str(tool), "rev-parse", "HEAD"], text=True).strip()
    if actual != TOOL_COMMIT:
        ap.error(f"tool commit {actual} != pinned {TOOL_COMMIT}")
    subprocess.run(["git", "-C", str(tool), "diff", "--quiet", "HEAD"], check=True)
    cfg = json.loads((model / "config.json").read_text())
    if cfg.get("model_type") != "deepseek_v41":
        ap.error("checkpoint is not deepseek_v41")
    if cfg.get("quantization_config", {}).get("expert_dtype") != "fp4":
        ap.error("expected official FP4 expert checkpoint")
    weights = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(weights.values()):
        if not (model / shard).is_file():
            ap.error(f"missing checkpoint shard: {shard}")
    subprocess.run([sys.executable, "-c",
                    "import pathlib,msmodelslim,sys; "
                    "assert pathlib.Path(msmodelslim.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]))",
                    str(tool)], check=True)
    work.mkdir(parents=True, exist_ok=False)
    template = (ROOT / "recipes/deepseek_v4_1_flash_w4a8.yaml").read_text()
    needle = "  dataset: mix_calib.jsonl"
    assert template.count(needle) == 1
    recipe.write_text(template.replace(needle, "  dataset: " + json.dumps(str(ROOT / "calib/mix_calib.jsonl"))))
    record = {
        "model": str(model), "tool_commit": actual, "devices": args.devices,
        "layers": args.layers, "commands": commands, "status": "started",
    }
    (work / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    for i, command in enumerate(commands):
        if i == 2:
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "configuration.json"):
                source = prepared / name
                if source.is_file() and not (output / name).exists():
                    shutil.copy2(source, output / name)
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=tool, check=True)
    record["status"] = "stage1-exported"
    (work / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Stage1 export completed: {output}; full-model assembly and accuracy validation remain.")


if __name__ == "__main__":
    main()
