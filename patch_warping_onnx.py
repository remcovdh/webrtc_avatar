#!/usr/bin/env python3
"""Upgrade FasterLivePortrait's volumetric GridSample model to ONNX opset 20."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, version_converter


TARGET_OPSET = 20


def _default_opset(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    raise RuntimeError("The model has no default ONNX opset import")


def _grid_sample_nodes(graph: onnx.GraphProto):
    for node in graph.node:
        if node.op_type == "GridSample" and node.domain in ("", "ai.onnx"):
            yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _grid_sample_nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child_graph in attribute.graphs:
                    yield from _grid_sample_nodes(child_graph)


def _normalize_grid_sample_modes(model: onnx.ModelProto) -> int:
    nodes = list(_grid_sample_nodes(model.graph))
    for node in nodes:
        for attribute in node.attribute:
            if attribute.name != "mode":
                continue
            if attribute.s == b"bilinear":
                attribute.s = b"linear"
            elif attribute.s == b"bicubic":
                attribute.s = b"cubic"
            if attribute.s not in (b"linear", b"nearest", b"cubic"):
                raise RuntimeError(
                    f"Unsupported GridSample mode after conversion: {attribute.s!r}"
                )
    return len(nodes)


def upgrade(model_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Warping model not found: {model_path}")

    marker_path = model_path.with_suffix(model_path.suffix + ".opset20")
    if (
        marker_path.is_file()
        and marker_path.stat().st_mtime_ns >= model_path.stat().st_mtime_ns
    ):
        print(f"[models] ONNX opset-20 marker is current: {model_path}")
        return

    print(f"[models] Inspecting volumetric warping model: {model_path}")
    model = onnx.load_model(model_path, load_external_data=True)
    source_opset = _default_opset(model)
    grid_sample_count = len(list(_grid_sample_nodes(model.graph)))
    if grid_sample_count == 0:
        raise RuntimeError(f"No GridSample nodes found in {model_path}")

    if source_opset < TARGET_OPSET:
        print(
            f"[models] Converting {grid_sample_count} GridSample node(s) "
            f"from ONNX opset {source_opset} to {TARGET_OPSET}"
        )
        model = version_converter.convert_version(model, TARGET_OPSET)

    converted_count = _normalize_grid_sample_modes(model)
    if _default_opset(model) < TARGET_OPSET:
        raise RuntimeError("ONNX version conversion did not reach opset 20")
    if converted_count != grid_sample_count:
        raise RuntimeError("GridSample node count changed during ONNX conversion")

    checker.check_model(model, full_check=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".opset20.tmp")
    try:
        onnx.save_model(model, temporary_path)
        checker.check_model(str(temporary_path), full_check=True)
        os.replace(temporary_path, model_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    marker_path.write_text(
        f"opset={_default_opset(model)}\ngrid_sample_nodes={converted_count}\n",
        encoding="utf-8",
    )
    print(f"[models] Volumetric ONNX model is ready: {model_path}")


def self_test() -> None:
    import onnxruntime as ort

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "GridSample",
                    ["input", "grid"],
                    ["output"],
                    mode="linear",
                    padding_mode="zeros",
                    align_corners=0,
                )
            ],
            "volumetric-grid-sample-smoke",
            [
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [1, 1, 2, 2, 2]
                ),
                helper.make_tensor_value_info(
                    "grid", TensorProto.FLOAT, [1, 1, 1, 1, 3]
                ),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 1, 1])],
        ),
        opset_imports=[helper.make_opsetid("", TARGET_OPSET)],
    )
    # The smoke graph only uses long-established protobuf fields; keeping its
    # IR version conservative avoids coupling the test to a newer ORT parser.
    model.ir_version = min(model.ir_version, 10)
    checker.check_model(model, full_check=True)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    result = session.run(
        None,
        {
            "input": np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2),
            "grid": np.zeros((1, 1, 1, 1, 3), dtype=np.float32),
        },
    )[0]
    if result.shape != (1, 1, 1, 1, 1):
        raise RuntimeError(f"Unexpected GridSample output shape: {result.shape}")
    print("ONNX Runtime volumetric GridSample smoke test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.model is not None:
        upgrade(args.model)
    else:
        parser.error("provide a model path or --self-test")


if __name__ == "__main__":
    main()
