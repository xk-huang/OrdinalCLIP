import argparse
import json
import os
import os.path as osp
import pdb
import random
from pathlib import Path
from collections import defaultdict
import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ordinalclip.runner.data import RegressionDataModule
from ordinalclip.runner.runner import Runner
from ordinalclip.utils.logging import get_logger, setup_file_handle_for_all_logger

matplotlib.use("agg")

logger = get_logger(__name__)


def main(cfg: DictConfig):
    pl.seed_everything(cfg.runner_cfg.seed, True)
    output_dir = Path(cfg.runner_cfg.output_dir)
    setup_file_handle_for_all_logger(str(output_dir / "vis.log"))
    deterministic = (
        False
        if cfg.runner_cfg.model_cfg.image_encoder_name.startswith("vgg")
        or cfg.runner_cfg.model_cfg.image_encoder_name.startswith("alex")
        else True
    )
    logger.info(f"`deterministic` flag: {deterministic}")

    if cfg.trainer_cfg.fast_dev_run is True:
        from IPython.core.debugger import set_trace

        set_trace()

    result_dict_ls = []
    # Visualizing
    runner = Runner(**OmegaConf.to_container(cfg.runner_cfg))
    with torch.no_grad():
        text_features = runner.module.forward_text_only()
        if len(text_features) == 101:
            text_features = text_features[15:76]  # morph range
            if cfg.num_one_side_long_tail > 0:
                num_tail = cfg.num_one_side_long_tail
                text_features = text_features[-num_tail:]
            print(f"len of proto: {len(text_features)}")
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = text_features @ text_features.t()
    logits_maxnorm = logits / logits.max()
    logits_softmax = logits.softmax(dim=-1)

    num_text_features = len(text_features)
    num_comparisons = num_text_features - 1
    ordinality_span_ls = [2 ** i for i in range(int(np.floor(np.log2(num_comparisons))) + 1)]
    ordinality_span_ls.append(num_comparisons)

    compare_sign_mat = torch.triu(logits_maxnorm[:-1, :-1] > logits_maxnorm[:-1, 1:])
    for span in ordinality_span_ls:
        mask = torch.ones_like(compare_sign_mat, dtype=torch.bool)
        mask_2 = ~torch.ones((num_comparisons - span, num_comparisons - span), dtype=torch.bool).triu()
        if num_comparisons - span > 0:
            mask[:(num_comparisons - span), -(num_comparisons - span):] &= mask_2
        mask = mask.triu()
        ordinality = ((compare_sign_mat * mask).sum() / mask.sum()).item()
        violate_ordinality = 1 - ordinality
        logger.info(f"span: {span} ordinality: {ordinality}, violate_ordinality: {violate_ordinality}")

        result_dict = dict(
                        name=str(output_dir),
                        model="before_optim",
                        ordinality=ordinality,
                        violate_ordinality=violate_ordinality,
                        span=span,
                    )
        result_dict_ls.append(result_dict)
        with open(f"{output_dir}/ordinality.json", "a") as f:
            f.write(
                json.dumps(
                    result_dict
                )
                + "\n"
            )

    plt.imshow(logits_maxnorm.cpu().numpy(), cmap="Reds")
    plt.colorbar()
    plt.savefig(f"{output_dir}/logits_maxnorm.before_optim.png", bbox_inches="tight")
    plt.clf()

    plt.imshow(logits_softmax.cpu().numpy(), cmap="Reds")
    plt.colorbar()
    plt.savefig(f"{output_dir}/logits_softmax.before_optim.png", bbox_inches="tight")
    plt.clf()

    ckpt_paths = list((output_dir / "ckpts").glob("*.ckpt"))
    for ckpt_path in ckpt_paths:
        logger.info(f"Start Visualizing ckpt: {ckpt_path}.")

        # no need to load weights in runner wrapper
        for k in cfg.runner_cfg.load_weights_cfg.keys():
            cfg.runner_cfg.load_weights_cfg[k] = None
        cfg.runner_cfg.ckpt_path = str(ckpt_path)

        runner = runner.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", **OmegaConf.to_container(cfg.runner_cfg)
        )
        with torch.no_grad():
            text_features = runner.module.forward_text_only()
            if len(text_features) == 101:
                text_features = text_features[15:76]  # morph range
                if cfg.num_one_side_long_tail > 0:
                    num_tail = cfg.num_one_side_long_tail
                    text_features = text_features[-num_tail:]
                print(f"len of proto: {len(text_features)}")
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = text_features @ text_features.t()
        logits_maxnorm = logits / logits.max()
        logits_softmax = logits.softmax(dim=-1)

        num_text_features = len(text_features)
        num_comparisons = num_text_features - 1
        ordinality_span_ls = [2 ** i for i in range(int(np.floor(np.log2(num_comparisons))) + 1)]
        ordinality_span_ls.append(num_comparisons)

        compare_sign_mat = torch.triu(logits_maxnorm[:-1, :-1] > logits_maxnorm[:-1, 1:])
        for span in ordinality_span_ls:
            mask = torch.ones_like(compare_sign_mat, dtype=torch.bool)
            mask_2 = ~torch.ones((num_comparisons - span, num_comparisons - span), dtype=torch.bool).triu()
            if num_comparisons - span > 0:
                mask[:(num_comparisons - span), -(num_comparisons - span):] &= mask_2
            mask = mask.triu()
            ordinality = ((compare_sign_mat * mask).sum() / mask.sum()).item()
            violate_ordinality = 1 - ordinality
            logger.info(f"span: {span} ordinality: {ordinality}, violate_ordinality: {violate_ordinality}")

            result_dict = dict(
                            name=str(output_dir),
                            model=str(ckpt_path.stem),
                            ordinality=ordinality,
                            violate_ordinality=violate_ordinality,
                            span=span,
                        )
            result_dict_ls.append(result_dict)
            with open(f"{output_dir}/ordinality.json", "a") as f:
                f.write(
                    json.dumps(
                        result_dict
                    )
                    + "\n"
                )

        plt.imshow(logits_maxnorm.cpu().numpy(), cmap="Reds")
        plt.colorbar()
        plt.savefig(f"{output_dir}/logits_maxnorm.{ckpt_path.stem}.png", bbox_inches="tight")
        plt.clf()

        plt.imshow(logits_softmax.cpu().numpy(), cmap="Reds")
        plt.colorbar()
        plt.savefig(f"{output_dir}/logits_softmax.{ckpt_path.stem}.png", bbox_inches="tight")
        plt.clf()

        logger.info(f"End Visualizing: {output_dir} .")

    new_result_dict = defaultdict(list)
    for result_dict_ in result_dict_ls:
        for key, val in result_dict_.items():
            new_result_dict[key].append(val)
    df = pd.DataFrame(new_result_dict)
    now = datetime.datetime.now().__format__("%y%m%d-%H%M%S")
    suffix = f".one_side_long_tail_{cfg.num_one_side_long_tail}" if cfg.num_one_side_long_tail > 0 else ""
    pth = f"{output_dir}/ordinality.{now}{suffix}.xlsx"
    df.to_excel(pth)
    logger.info(f"End : {pth}.")

def setup_output_dir_for_training(output_dir):
    output_dir = Path(output_dir)

    if output_dir.stem.startswith("version_"):
        output_dir = output_dir.parent
    output_dir = output_dir / f"version_{get_version(output_dir)}"

    return output_dir


def get_version(path: Path):
    versions = path.glob("version_*")
    return len(list(versions))


def parse_cfg(args, instantialize_output_dir=True):
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    extra_cfg = OmegaConf.from_dotlist(args.cfg_options)
    cfg = OmegaConf.merge(cfg, extra_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.create())

    # Setup output_dir
    output_dir = Path(cfg.runner_cfg.output_dir if args.output_dir is None else args.output_dir)
    if instantialize_output_dir:
        if not args.test_only:
            output_dir = setup_output_dir_for_training(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    cli_cfg = OmegaConf.create(
        dict(
            config=args.config,
            test_only=args.test_only,
            runner_cfg=dict(seed=args.seed, output_dir=str(output_dir)),
            trainer_cfg=dict(fast_dev_run=args.debug),
        )
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)
    if instantialize_output_dir:
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", type=str, default=[])
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--num_one_side_long_tail", type=int, default=0)
    parser.add_argument(
        "--cfg_options",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg = parse_cfg(args, instantialize_output_dir=False)
    cfg.num_one_side_long_tail = args.num_one_side_long_tail
    logger.info("Start.")
    main(cfg)
    logger.info("End.")
