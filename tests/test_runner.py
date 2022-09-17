import os.path as osp

import torch
from omegaconf import OmegaConf

from ordinalclip.models.ordinalclip import OrdinalCLIP
from ordinalclip.runner.runner import Runner


def test_clipper():
    cfg = OmegaConf.load(osp.join(osp.dirname(__file__), "data", "default.yaml"))
    runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
    BS = 3
    imgs = torch.randn(
        BS, 3, runner.module.image_encoder.input_resolution, runner.module.image_encoder.input_resolution
    )
    output = runner(imgs)
