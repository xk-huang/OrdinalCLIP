import os.path as osp

import torch.nn as nn

from ordinalclip.utils import get_logger

from .builder import MODELS
from .ordinalclip import load_clip_to_cpu

logger = get_logger(__name__)


@MODELS.register_module()
class Baseline(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        self.image_encoder = clip_model.visual
        self.text_encoder = None
        self.prompt_learner = None
        self.logit_scale = None

        self.embed_dims = clip_model.text_projection.shape[1]

        self.num_ranks = prompt_learner_cfg["num_ranks"]
        self.last_project = nn.Linear(self.embed_dims, self.num_ranks, bias=False)

    def forward(self, images):
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.forward_text_only()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return 100.0 * image_features @ text_features.t(), None, None

    def forward_text_only(self):
        text_features = self.last_project.weight

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)