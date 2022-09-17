import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
from glob import glob
from itertools import product
from pathlib import Path

from omegaconf import OmegaConf


def save_exps(args, generated_config_dir, output_base_dir, name, raw_perp_yaml_list):
    add_to_name, perp_yaml_list = prepare_yaml_list(args, raw_perp_yaml_list)

    process_ls = []
    for yamls in product(*perp_yaml_list):

        cli_list = ["python", f"{osp.dirname(__file__)}/config_generator.py"]
        exp_name = [name]
        for i, yaml in enumerate(yamls):
            cli_list += ["--config", yaml]
            if add_to_name[i] is True:
                yaml = Path(yaml)
                exp_name.append(yaml.stem)

        exp_name = "_".join(exp_name)
        cli_list += ["--output_dir", str(output_base_dir / exp_name)]
        cli_list += ["--dump_cfg_path", str(generated_config_dir / f"{exp_name}.yaml")]

        process_ls.append(subprocess.Popen(cli_list))
    return process_ls


def prepare_yaml_list(args, raw_perp_yaml_list):
    add_to_name = []
    perp_yaml_list = []
    for yaml_list in raw_perp_yaml_list:
        if osp.isfile(yaml_list) and yaml_list.endswith(".yaml"):
            perp_yaml_list.append([yaml_list])
            add_to_name.append(True if args.add_constant_config_name else False)
        elif osp.isdir(yaml_list):
            add_to_name.append(True)
            perp_yaml_list.append(list(glob(osp.join(yaml_list, "*.yaml"))))
        else:
            raise TypeError(f"Werid {yaml_list}, in {args.meta_config}")
    return add_to_name, perp_yaml_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_config", "-c", type=str)
    parser.add_argument("--generated_config_base_dir", type=str, default="configs/sweep_cfgs")
    parser.add_argument("--output_base_dir", type=str, default="results")
    parser.add_argument("--add_constant_config_name", action="store_true")
    args = parser.parse_args()

    configs = OmegaConf.load(args.meta_config)
    meta_config_name = Path(args.meta_config).stem
    generated_config_base_dir = Path(args.generated_config_base_dir) / meta_config_name
    output_base_dir = Path(args.output_base_dir) / meta_config_name
    if generated_config_base_dir.exists():
        shutil.rmtree(generated_config_base_dir)
    generated_config_base_dir.mkdir(parents=True, exist_ok=True)

    process_ls = []
    for name, raw_perp_yaml_list in configs.items():
        if raw_perp_yaml_list is None:
            print(f"empty: {name}")
            continue
        else:
            print(f"generating: {name}")
        process_ls += save_exps(args, generated_config_base_dir, output_base_dir, name, raw_perp_yaml_list)

    print(f"num of exp: {len(process_ls)}")
    for process in process_ls:
        process.wait()
