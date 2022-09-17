import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

sys.path.append(str(Path(__file__).parent.parent))
from run import parse_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", action="append", type=str, default=[])
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--dump_cfg_path", default=None, type=str)
    parser.add_argument(
        "--cfg_options",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    cfg: DictConfig = parse_cfg(args, instantialize_output_dir=False)

    if args.dump_cfg_path is not None:
        dump_cfg_path = Path(args.dump_cfg_path)
        OmegaConf.save(cfg, args.dump_cfg_path)
    else:
        print(OmegaConf.to_yaml(cfg))
