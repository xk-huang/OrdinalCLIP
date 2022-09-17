import os.path as osp
from itertools import product

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from ordinalclip.models.ordinalclip import OrdinalCLIP, load_clip_to_cpu
from ordinalclip.models.prompt_leaners import PROMPT_LEARNERS
from ordinalclip.models.prompt_leaners.rank_prompt_learner import RankPromptLearner


def test_clipper():
    num_ranks = 10
    num_base_ranks = 5
    num_tokens_per_rank = 2
    num_context_tokens = 5
    interpolation_type = "linear"
    init_rank_path = None
    rank_tokens_position = "tail"
    init_context = None
    rank_specific_context = False

    cfg = OmegaConf.create(
        dict(
            prompt_learner_cfg=dict(
                type="RankPromptLearner",
                num_ranks=num_ranks,
                num_base_ranks=num_base_ranks,
                num_tokens_per_rank=num_tokens_per_rank,
                num_context_tokens=num_context_tokens,
                rank_tokens_position=rank_tokens_position,
                init_rank_path=init_rank_path,
                init_context=init_context,
                rank_specific_context=rank_specific_context,
                interpolation_type=interpolation_type,
            ),
            text_encoder_name="RN50",
            image_encoder_name="vgg16",
        )
    )
    OmegaConf.save(cfg, "clipper.yaml")
    model = OrdinalCLIP(**OmegaConf.to_container(cfg))

    BS = 3
    imgs = torch.randn(BS, 3, model.image_encoder.input_resolution, model.image_encoder.input_resolution)
    output = model(imgs)
