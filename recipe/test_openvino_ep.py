"""Regression guard for the OpenVINO execution provider.

onnxruntime links the conda-forge ``libonnx`` (which registers
``onnx/onnx-ml.proto`` into the process-global protobuf descriptor pool),
while OpenVINO's ``libopenvino-onnx-frontend`` vendors its own copy of onnx
and, being built against the *shared* system libprotobuf, registers the same
``onnx/onnx-ml.proto`` file into that same pool. When both end up in one
process protobuf aborts the process:

    [libprotobuf FATAL] File already exists in database: onnx/onnx-ml.proto
    Check failed: GeneratedDatabase()->Add(encoded_file_descriptor, size)
    -> SIGABRT

A bare ``get_available_providers()`` check does NOT catch this: the provider
is advertised lazily and the OpenVINO libraries are only dlopen'd when an
OpenVINO session is actually constructed. This test therefore *constructs* an
OpenVINO EP session so the OpenVINO libraries are loaded in-process alongside
onnxruntime's ONNX/protobuf -- exactly the path that crashes on a descriptor
collision. If the collision regresses, this test aborts (non-zero exit) and
the build fails instead of silently shipping a driver that crashes the first
time a user selects OpenVINOExecutionProvider.
"""

import sys

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

EP = "OpenVINOExecutionProvider"

providers = ort.get_available_providers()
assert EP in providers, f"{EP} not advertised; available: {providers}"

# Minimal Relu model so the session has something real to run.
node = helper.make_node("Relu", ["x"], ["y"])
graph = helper.make_graph(
    [node],
    "relu_graph",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

# Constructing the session is the step that dlopens the OpenVINO libraries
# (libopenvino + libopenvino-onnx-frontend) into this process. On a protobuf
# onnx-ml.proto descriptor collision this aborts with SIGABRT before returning.
session = ort.InferenceSession(
    model.SerializeToString(),
    providers=[EP, "CPUExecutionProvider"],
)

# Sanity-check the math and confirm OpenVINO actually serviced the session
# (rather than silently falling back to CPU, which would hide a broken EP).
out = session.run(None, {"x": np.array([-1.0, 0.0, 2.0], dtype=np.float32)})[0]
np.testing.assert_allclose(out, [0.0, 0.0, 2.0])
assert EP in session.get_providers(), session.get_providers()

print("OpenVINO EP session created and ran; onnx-ml.proto collision guard passed")
sys.exit(0)
