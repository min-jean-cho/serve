import logging
import os
import importlib.util
import time
import torch
import subprocess
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

#assert os.environ.get("TS_IPEX_ENABLE", "false") == "true", "Please make sure IPEX is enabled for IPEXHandler"

try:
    import intel_extension_for_pytorch as ipex
    ipex_enabled = True
except ImportError as error:
    logger.error("IPEX is enabled but intel-extension-for-pytorch is not installed. Please install IPEX")
    exit(-1)

class IPEXHandler(BaseHandler):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """

    def __init__(self):
        super().__init__()

    def initialize(self, context):
        super().initialize(context)
        
        if os.environ.get("TS_IPEX_CHANNEl_LAST", "true") == "true":
            self.model = self.model.to(memory_format=torch.channels_last) 
        
        is_bf16_supported_hw = self.is_bf16_supported()
        if os.environ.get("TS_IPEX_DTYPE", "float32") == "bfloat16" and not is_bf16_supported_hw:
            os.environ["TS_IPEX_DTYPE"] = "float32"
            logger.info("You have specified bfloat16 dtype, but bfloat16 dot-product hardware accelerator is not supported in your current hardware. Proceeding with float32 dtype instead.")

        if os.environ.get("TS_IPEX_DTYPE", "float32") == "float32":
            self.model = ipex.optimize(self.model, dtype=torch.float32)
        elif os.environ.get("TS_IPEX_DTYPE", "float32") == "bfloat16":
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
        
        
        if os.environ.get("TS_IPEX_MODE", "imperative") == "torchscript":
            if os.environ.get("TS_IPEX_INPUT_TENSOR_SHAPE", "null") == "null":
                logger.debug("Please specify valid input tensor shape for torchscript mode.")
            else:
                jit_inputs = self.convert_input_tensor_shape_to_jit_inputs()
                
            if os.environ.get("TS_IPEX_DTYPE", "float32") == "float32":
                with torch.no_grad():
                    self.model = torch.jit.trace(self.model, jit_inputs)
                    self.model = torch.jit.freeze(self.model)
            elif os.environ.get("TS_IPEX_DTYPE", "float32") == "bfloat16":
                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                    with torch.no_grad():
                        self.model = torch.jit.trace(self.model, jit_inputs)
                        self.model = torch.jit.freeze(self.model)
                        
    
    def inference(self, data, *args, **kwargs):
        if os.environ.get("TS_IPEX_CHANNEl_LAST", "true") == "true":
            data = data.contiguous(memory_format=torch.channels_last)
        marshalled_data = data.to(self.device)
        
        if os.environ.get("TS_IPEX_DTYPE", "float32") == "bfloat16":
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                with torch.no_grad():
                    results = self.model(marshalled_data, *args, **kwargs)
        else:
            with torch.no_grad():
                results = self.model(marshalled_data, *args, **kwargs)
                
        return results
    
    def convert_input_tensor_shape_to_jit_inputs(self):
        input_tensor_shape = os.environ.get("TS_IPEX_INPUT_TENSOR_SHAPE")
        input_tensor_shape = list(input_tensor_shape.split(";"))
        jit_inputs = []
        
        channel_last = False 
        if os.environ.get("TS_IPEX_CHANNEl_LAST", "true") == "true":
            channel_last = True
            
        for _ in input_tensor_shape:
            jit_input_shape = tuple(int(x) for x in _.split(","))
            jit_input = torch.randn(jit_input_shape)
            if channel_last:
                jit_input = jit_input.contiguous(memory_format=torch.channels_last)
            jit_inputs.append(jit_input)
        jit_inputs = tuple(jit_inputs)
        return jit_inputs 
    
    def is_bf16_supported(self):
        proc1 = subprocess.Popen(['lscpu'], stdout=subprocess.PIPE)
        proc2 = subprocess.Popen(['grep', 'Flags'], stdin=proc1.stdout, stdout=subprocess.PIPE)
        proc1.stdout.close()
        out = proc2.communicate()
        return 'bf16' in str(out)
        
        