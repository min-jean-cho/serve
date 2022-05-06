import os
import torch
import torch.fx.experimental.optimization as optimization
import logging
import intel_extension_for_pytorch as ipex
logger = logging.getLogger(__name__)

TORCH_DTYPE = {
    "TYPE_FP16": torch.float16,
    "TYPE_FP32": torch.float32,
    "TYPE_FP64": torch.float64,

    "TYPE_BF16": torch.bfloat16,

    "TYPE_UINT8": torch.uint8,

    "TYPE_INT8": 	torch.int8,
    "TYPE_INT16": torch.int16,
    "TYPE_INT32": torch.int32,
    "TYPE_INT64": torch.int64
}

TORCH_QSCHEME = {
    "per_tensor_affine": torch.per_tensor_affine,
    "per_tensor_symmetric": torch.per_tensor_symmetric
}


def prepare_ipex_optimized_model(model, ipex_dtype, ipex_mode, ipex_channel_last, input_tensor_shapes, input_tensor_dtype, ipex_conv_bn_folding, ipex_qscheme):
    if ipex_channel_last == "true":
        model = model.to(memory_format=torch.channels_last)

    # ipex optimize
    if ipex_dtype == "float32":
        model = ipex.optimize(model, dtype=torch.float32)
    elif ipex_dtype == "bfloat16":
        model = ipex.optimize(model, dtype=torch.bfloat16)

    # torchscript
    if ipex_mode == "torchscript":
        x = []
        for input_tensor_shape in input_tensor_shapes.split(";"):
            input_tensor_shape = tuple(int(_)
                                       for _ in input_tensor_shape.split(","))
            dummy_tensor = torch.ones(
                input_tensor_shape, dtype=TORCH_DTYPE[input_tensor_dtype])
            if ipex_channel_last == "true":
                dummy_tensor = dummy_tensor.contiguous(
                    memory_format=torch.channels_last)
            x.append(dummy_tensor)
        
        if ipex_dtype == "float32":
            with torch.no_grad():
                model = torch.jit.trace(model, *x)
                model = torch.jit.freeze(model)
        elif ipex_dtype == "bfloat16":
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
                model = torch.jit.trace(model, *x)
                model = torch.jit.freeze(model)
        elif ipex_dtype == "int8":
            if ipex_conv_bn_folding == "true":
                model = optimization.fuse(model)

            # calibration
            conf = ipex.quantization.QuantConf(
                qscheme=TORCH_QSCHEME[ipex_qscheme])
            n_iter = 1
            with torch.no_grad():
                for i in range(n_iter):
                    with ipex.quantization.calibrate(conf):
                        model(*x)

            # conversion
            model = ipex.quantization.convert(model, conf, x)
    return model
