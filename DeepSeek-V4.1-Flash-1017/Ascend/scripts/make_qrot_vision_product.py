#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_qrot_vision_product.py -- fold the QuaRot global rotation into the
vision output boundary of a W4A8 product.

Why: the W4A8 checkpoint stores ``embed.weight`` already rotated
(``E_prod == E_official @ R``), so the LLM residual stream lives in the rotated
basis.  The product was assembled with ``--vision drop`` and the vision boundary
was re-added verbatim from the official checkpoint, i.e. still in the
*unrotated* basis.  Official Aurora W8A8 exports avoid this by listing
``aligner.w2.weight``, ``aligner.w2.bias``, ``image_start``, ``image_end``,
``image_newline`` under QuaRot ``left_global``.

Math (PyTorch ``nn.Linear``: ``y = x @ W.T + b``, ``W`` is ``[out, in]``).
The official-basis output row ``y`` must become ``y @ R`` to match the rotated
residual stream, hence ``W' = R.T @ W`` and ``b' = b @ R``.

Usage:
    python3 make_qrot_vision_product.py --src <product> --dst <new product> [--dry-run]
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD = "vision-00001-of-00001.safetensors"
WEIGHT_KEYS = ("aligner.w2.weight",)
ROW_KEYS = ("aligner.w2.bias", "image_start", "image_end", "image_newline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--direction", choices=("R", "RT"), default="R",
                    help="R: row-major, rows become v @ R (matches embed.weight); "
                         "RT: the transpose fallback, rows become v @ R.T")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with safe_open(os.path.join(args.src, "optional", "quarot.safetensors"),
                   framework="pt", device="cpu") as fh:
        rot = fh.get_tensor("global_rotation").float()
    if args.direction == "RT":
        rot = rot.T.contiguous()
    hidden = rot.shape[0]
    print("[qrot] global_rotation %s direction=%s" % (tuple(rot.shape), args.direction))

    if not args.dry_run:
        os.makedirs(args.dst, exist_ok=True)
        for name in os.listdir(args.src):
            if name == "optional":
                continue
            s = os.path.join(args.src, name)
            d = os.path.join(args.dst, name)
            if os.path.islink(s):
                target = os.readlink(s)
                if os.path.islink(d) or os.path.exists(d):
                    os.remove(d)
                os.symlink(target, d)
            elif os.path.isfile(s):
                shutil.copy2(s, d)
        opt_src = os.path.join(args.src, "optional")
        opt_dst = os.path.join(args.dst, "optional")
        if os.path.isdir(opt_src):
            shutil.copytree(opt_src, opt_dst, dirs_exist_ok=True)

    src_shard = os.path.join(args.src, SHARD)
    with safe_open(src_shard, framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
        tensors = {k: fh.get_tensor(k) for k in keys}

    report = {}
    for key in WEIGHT_KEYS:
        if key not in tensors:
            continue
        w = tensors[key]
        assert w.dim() == 2 and w.shape[0] == hidden, (key, tuple(w.shape))
        orig_dtype = w.dtype
        rotated = (rot.T @ w.float()).to(orig_dtype).contiguous()
        report[key] = {"shape": list(w.shape), "dtype": str(orig_dtype),
                       "changed": not torch.equal(rotated, w)}
        tensors[key] = rotated
        print("[qrot] %-22s %s -> R.T @ W" % (key, tuple(w.shape)))

    for key in ROW_KEYS:
        if key not in tensors:
            continue
        v = tensors[key]
        assert v.dim() == 1 and v.shape[0] == hidden, (key, tuple(v.shape))
        orig_dtype = v.dtype
        rotated = (v.float() @ rot).to(orig_dtype).contiguous()
        report[key] = {"shape": list(v.shape), "dtype": str(orig_dtype),
                       "changed": not torch.equal(rotated, v)}
        tensors[key] = rotated
        print("[qrot] %-22s %s -> v @ R" % (key, tuple(v.shape)))

    if not args.dry_run:
        out_shard = os.path.join(args.dst, SHARD)
        save_file(tensors, out_shard, metadata={"format": "pt"})
        print("[qrot] wrote %s (%d tensors, %.3f GB)" % (
            out_shard, len(tensors), os.path.getsize(out_shard) / 1e9))
        with open(os.path.join(args.dst, "qrot_vision_report.json"), "w", encoding="utf-8") as fh:
            json.dump({"src": args.src, "dst": args.dst, "rotation": list(rot.shape),
                       "direction": args.direction,
                       "folded": report}, fh, ensure_ascii=False, indent=1)

    for k, v in report.items():
        print("[qrot] %-22s changed=%s" % (k, v["changed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
