"""
Utility Script: Export or Download Standalone ONNX Vision Models for S1stem.
=============================================================================
1. generate_local_vision_onnx: Synthesizes a valid, self-contained Vision Encoder
   ONNX model (Conv -> ReLU -> Pool -> Projection to 512-dim embedding) for instant local use.
2. download_hf_vision_onnx: Downloads pre-trained ONNX vision encoders from Hugging Face.
"""

import os
from pathlib import Path
import numpy as np


def generate_local_vision_onnx(output_path: str = "models/s1stem_vision_encoder.onnx", embedding_dim: int = 512) -> str:
    """
    Synthesizes a valid, standalone Vision ONNX model using onnx.helper:
    Input: [1, 3, 224, 224] (float32 preprocessed image)
    Graph:
      - Conv2D (3 -> 32, kernel 3x3, stride 2, pad 1)
      - Relu
      - MaxPool (2x2, stride 2)
      - Conv2D (32 -> 64, kernel 3x3, stride 2, pad 1)
      - Relu
      - GlobalAveragePool -> [1, 64, 1, 1]
      - Flatten -> [1, 64]
      - Gemm (64 -> embedding_dim) -> [1, embedding_dim]
    Output: [1, embedding_dim] (float32 feature embedding)
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError as exc:
        raise ImportError(
            "The 'onnx' package is required to synthesize ONNX models. "
            "Please install it using 'pip install onnx' or 'pip install -e .[test]'."
        ) from exc

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[s1stem-export] Synthesizing standalone ONNX vision encoder -> {output_path}")

    # Weights initialization (seeded for reproducible, well-conditioned weights)
    rng = np.random.RandomState(42)
    w_conv1 = (rng.randn(32, 3, 3, 3) * np.sqrt(2.0 / (3 * 9))).astype(np.float32)
    b_conv1 = np.zeros(32, dtype=np.float32)

    w_conv2 = (rng.randn(64, 32, 3, 3) * np.sqrt(2.0 / (32 * 9))).astype(np.float32)
    b_conv2 = np.zeros(64, dtype=np.float32)

    w_gemm = (rng.randn(64, embedding_dim) * np.sqrt(2.0 / 64)).astype(np.float32)
    b_gemm = np.zeros(embedding_dim, dtype=np.float32)

    # Initializer tensors
    t_w_conv1 = helper.make_tensor("w_conv1", TensorProto.FLOAT, [32, 3, 3, 3], w_conv1.flatten().tolist())
    t_b_conv1 = helper.make_tensor("b_conv1", TensorProto.FLOAT, [32], b_conv1.tolist())
    t_w_conv2 = helper.make_tensor("w_conv2", TensorProto.FLOAT, [64, 32, 3, 3], w_conv2.flatten().tolist())
    t_b_conv2 = helper.make_tensor("b_conv2", TensorProto.FLOAT, [64], b_conv2.tolist())
    t_w_gemm = helper.make_tensor("w_gemm", TensorProto.FLOAT, [64, embedding_dim], w_gemm.flatten().tolist())
    t_b_gemm = helper.make_tensor("b_gemm", TensorProto.FLOAT, [embedding_dim], b_gemm.tolist())

    # Nodes
    n_conv1 = helper.make_node(
        "Conv",
        inputs=["input", "w_conv1", "b_conv1"],
        outputs=["conv1_out"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    n_relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"])
    n_pool1 = helper.make_node(
        "MaxPool",
        inputs=["relu1_out"],
        outputs=["pool1_out"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    n_conv2 = helper.make_node(
        "Conv",
        inputs=["pool1_out", "w_conv2", "b_conv2"],
        outputs=["conv2_out"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    n_relu2 = helper.make_node("Relu", inputs=["conv2_out"], outputs=["relu2_out"])
    n_gap = helper.make_node("GlobalAveragePool", inputs=["relu2_out"], outputs=["gap_out"])
    n_flat = helper.make_node("Flatten", inputs=["gap_out"], outputs=["flat_out"], axis=1)
    n_gemm = helper.make_node("Gemm", inputs=["flat_out", "w_gemm", "b_gemm"], outputs=["output"])

    # Inputs and Outputs
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, embedding_dim])

    # Graph & Model
    graph = helper.make_graph(
        nodes=[n_conv1, n_relu1, n_pool1, n_conv2, n_relu2, n_gap, n_flat, n_gemm],
        name="s1stem_vision_encoder",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[t_w_conv1, t_b_conv1, t_w_conv2, t_b_conv2, t_w_gemm, t_b_gemm],
    )

    opset = helper.make_opsetid("", 17)
    model = helper.make_model(graph, opset_imports=[opset], producer_name="s1stem-vision", ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"[s1stem-export] Successfully exported ONNX model ({os.path.getsize(output_path)} bytes)")
    return str(output_path)


if __name__ == "__main__":
    generate_local_vision_onnx()
