import argparse
import datetime
import json
import os.path as osp
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_exp_name(stats_dict):
    exp_name = []

    output_dir = stats_dict.pop("output_dir", None)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.stem.startswith("version_"):
            exp_name += [output_dir.parent.stem, output_dir.stem.replace("_", "-")]
        else:
            exp_name.append(output_dir.stem)

    ckpt_name = stats_dict.pop("ckpt_path", None)
    if ckpt_name is not None and ckpt_name != "":
        ckpt_name = Path(ckpt_name)
        exp_name.append(ckpt_name.stem.replace("_", "-"))

    exp_name = "_".join(exp_name)
    try:
        head, tail = re.split("fold-[0-9]+", exp_name, 1)
        mid = re.findall("fold-[0-9]+", exp_name)[0]
        assert head + mid + tail == exp_name
        if len(head) > 0 and head[-1] == "-":
            head = head[:-1] + "_"
        if len(tail) > 0 and tail[0] == "-":
            tail = "_" + tail[1:]
        exp_name = head + mid + tail
    except IndexError:
        pass
    except ValueError:
        pass
    stats_dict["exp_name"] = exp_name


def parse_one_stats_file(stats_path):
    stats_dict_ls = []
    num_exp_name_fields = []
    if not osp.isfile(stats_path):
        print(f"ignore: {stats_path}")
        return

    print(f"stats_file: {stats_path}")
    with open(stats_path, "r") as f:
        for line in f.readlines():
            stats_dict = json.loads(line)
            prepare_exp_name(stats_dict)

            stats_dict_ls.append(stats_dict)
            num_exp_name_fields.append(len(stats_dict["exp_name"].split("_")))

    for stats_dict in stats_dict_ls:
        exp_name = stats_dict.pop("exp_name")
        exp_name_per_fields = exp_name.split("_")

        exp_name_fields_id = 1
        for exp_name_one_field in exp_name_per_fields:
            if re.match("^fold-[0-9]+$", exp_name_one_field):
                stats_dict["fold"] = exp_name_one_field
            else:
                stats_dict[f"field-{exp_name_fields_id}"] = exp_name_one_field
                exp_name_fields_id += 1

    return pd.DataFrame(stats_dict_ls)


def parse_stats_file_dir(stats_file_dir, json_file_name="test_stats.json"):
    cli_find = f"find {stats_file_dir} -name '{json_file_name}'"
    print(cli_find)
    stats_file_ls = subprocess.check_output(cli_find, shell=True).decode(sys.stdout.encoding).strip().split("\n")
    stats_df_ls = []
    for stats_file in stats_file_ls:
        stats_df_ls.append(parse_one_stats_file(stats_file))

    return pd.concat(stats_df_ls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", "-d", type=str, default="results/")
    parser.add_argument("--stdin", "-T", type=argparse.FileType("r"))
    parser.add_argument("--output_dir", "-o", type=str, default="results/stats/")
    parser.add_argument("--output_name", "-n", type=str, default=None)
    parser.add_argument("--parsed_file", "-p", type=str, default="test_stats.json")
    parser.add_argument("--output_type", "-t", default="csv", choices=("csv", "xlsx"))

    args = parser.parse_args()
    json_file_name = args.parsed_file
    try:
        if args.stdin is not None:
            print(f"parse from: {args.stdin}")
            stats = pd.concat(
                [parse_stats_file_dir(_dir.strip(), json_file_name=json_file_name) for _dir in args.stdin]
            )
        else:
            print(f"parse from: {args.results_dir}")
            stats = parse_stats_file_dir(args.results_dir, json_file_name=json_file_name)
    except ValueError:
        print(
            f"(one of) the folds under `{args.results_dir}` / `{args.stdin}` does not contain `val/test_stats.json` file."
        )
        exit()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name is None:
        args.output_name = Path(args.results_dir).name

    now = datetime.datetime.now().__format__("%y%m%d-%H%M%S")
    if args.output_type == "csv":
        file_path = str(output_dir / f"{args.output_name}.{now}.csv")
        stats.to_csv(file_path, index=False)
        print(f"write to: {file_path}")
    elif args.output_type == "xlsx":
        file_path = str(output_dir / f"{args.output_name}.{now}.xlsx")
        stats.to_excel(file_path, index=False)
        print(f"write to: {file_path}")
