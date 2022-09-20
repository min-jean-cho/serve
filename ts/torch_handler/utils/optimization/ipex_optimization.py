from .optimization import optimization_registry, Optimization
from ..conf.ipex_config import Conf
import os
import torch
import logging
import subprocess 

logger = logging.getLogger(__name__)

@optimization_registry
class IPEXOptimization(Optimization):
    """The Intel® Extension for PyTorch* (IPEX) Optimization.
    
    Args:
        cfg (Conf): the optimization configuration.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.dtype = cfg.dtype
        self.channels_last = cfg.channels_last
        
        if self.dtype == 'int8':
            self.quantization_approach = cfg.quantization.approach
            self.quantization_example_inputs = cfg.quantization.example_inputs
            self.quantization_calibration_dataset = cfg.quantization.calibration_dataset
        
        self.torchscript = False 
        if cfg.torchscript.approach is not None:
            self.torchscript = True 
            self.torchscript_approach = cfg.torchscript.approach
            self.torchscript_example_inputs = cfg.torchscript.example_inputs

    def optimize(self, model):
        """Apply Intel® Extension for PyTorch* (IPEX) optimizations to the given model (nn.Module).
           
        Args:
            model (torch.nn.Module): The model to optimize. 
        Returns:
            torch.nn.Module: The optimized model.
        """
        import intel_extension_for_pytorch as ipex
        
        # channel last 
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # dtype 
        if self.dtype == 'float32':
            model = ipex.optimize(model, dtype=torch.float32)
        elif self.dtype == 'bfloat16':
            model = ipex.optimize(model, dtype=torch.bfloat16)
        else: # int8
            from intel_extension_for_pytorch.quantization import prepare, convert
            
            if self.quantization_approach == 'static':
                qconfig = ipex.quantization.default_static_qconfig
                
                # prepare 
                model = prepare(model, qconfig, example_inputs=self.quantization_example_inputs, inplace=False)
                
                # calibrate
                for x in self.quantization_calibration_dataset:
                    model(*x)
            else: # dynamic
                qconfig = ipex.quantization.default_dynamic_qconfig
                
                # prepare
                model = prepare(model, qconfig, example_inputs=self.quantization_example_inputs)
            
            # convert 
            model = convert(model)
        
        # torchscript 
        if self.torchscript:
            with torch.cpu.amp.autocast(enabled=self.dtype=='bfloat16'), torch.no_grad():
                if self.torchscript_approach == 'trace':
                    try:
                        model = torch.jit.trace(model, example_inputs=self.torchscript_example_inputs)
                    except:
                        try: 
                            model = torch.jit.trace(model, example_inputs=self.torchscript_example_inputs, check_trace=False, strict=False)
                        except:
                            logger.error("TorchScript tracing the model failed. Make sure the model is traceable.")
                            exit(-1)
                model = torch.jit.freeze(model)
        
        return model