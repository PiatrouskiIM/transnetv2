import torch
import coremltools as ct
import coremltools.optimize.coreml as cto
from ..utils import inference_format_by_extension


def save(trans_net_v2_ane,
         save_path: str,
         convert_to="mlprogram",
         minimum_deployment_target=ct.target.iOS16,
         compute_precision=ct.precision.FLOAT16,
         inputs=(ct.TensorType(name="input", shape=(1, 3, 100, 27, 48)),),
         compute_units=ct.ComputeUnit.CPU_AND_NE) -> None:
    assert not trans_net_v2_ane.training, """To export a model to onnx format, the model must be in evaluation """ \
                                          """mode. Consider calling .eval() before running this function."""
    assert inference_format_by_extension(save_path) == "onnx", "Wrong save_path."
    x = torch.randint(low=0, high=255, size=(1, 3, 100, 27, 48), dtype=torch.uint8)
    traced_model = torch.jit.trace(trans_net_v2_ane, x)
    ct_model = ct.convert(traced_model,
                          inputs=inputs,
                          convert_to=convert_to,
                          compute_units=compute_units,
                          compute_precision=compute_precision,
                          minimum_deployment_target=minimum_deployment_target)
    ct_model.save(save_path)


def quantize(source_path, target_path):
    model = ct.models.MLModel(source_path)
    op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
    config = cto.OptimizationConfig(global_config=op_config)
    compressed_8_bit_model = cto.linear_quantize_weights(model, config=config)
    compressed_8_bit_model.save(target_path)
