"""
Module for image classification default handler
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
import logging
import os

from .vision_handler import VisionHandler
from ..utils.util  import map_class_to_label

logger = logging.getLogger(__name__)

ipex_enabled = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
    ipex_enabled = True
    try:
        import accimage
        ipex_enabled = True
    except ImportError as error:
        logger.warning("Please install accimage for optimized preprocessing")
 


class ImageClassifier(VisionHandler):
    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """

    topk = 5
    # These are the standard Imagenet dimensions
    # and statistics
    
    if not ipex_enabled:
        image_processing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    else:
        mean_ = [[[0.4850]],[[0.4560]],[[0.4060]]]
        mean_ = torch.as_tensor(mean_)
        
        std_ = [[[0.229]],[[0.224]],[[0.225]]]
        std_ = torch.as_tensor(std_)
        
        
        image_processing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_,std=std_, inplace=True)
        ])

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):
        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return map_class_to_label(probs, self.mapping, classes)
