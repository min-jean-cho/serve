import os
import logging
from abc import ABC

import torch

from ts.context import Context
#from ts.torch_handler.base_handler import BaseHandler
from ts.torch_handler.image_classifier import ImageClassifier

logger = logging.getLogger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as error:
    logger.error(
        "intel-extension-for-pytorch is not installed"
    )
    raise error

class AMXHandler(ImageClassifier, ABC):
    """
    Base Intel® Advanced Matrix Extensions (Intel® AMX) handler to 
    accelerate on 4th Gen Intel® Xeon® processors
    """
    
    def __init__(self):
        super().__init__()
        
        self.amx_dtype = torch.bfloat16
        self.amp_enabled = True

    def initialize(self, ctx: Context):
        handler_config = ctx.model_yaml_config.get("handler", {})
        properties = ctx.system_properties
        
        # Intel AMX BF16
        if handler_config.get("dtype", "bfloat16") == "bfloat16":
          self.amx_dtype = torch.bfloat16
          self.amp_enabled = True
        # TODO: Intel AMX INT8
          
        self.manifest = ctx.manifest

        model_dir = properties.get("model_dir")
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(
                model_dir, model_file, self.model_pt_path
            )
            self.model.eval()
        
        self.model = ipex.optimize(self.model, dtype=self.amx_dtype)
        
        logger.info("Successfully loaded Model %s", ctx.model_name)
        self.initialized = True
        
    def inference(self, data, *args, **kwargs):
        marshalled_data = data
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amx_dtype if self.amp_enabled else None,):
            results = self.model(marshalled_data, *args, **kwargs)
        return results
        