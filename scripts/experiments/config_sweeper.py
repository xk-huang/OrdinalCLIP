import argparse
import collections
import datetime
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import List

from omegaconf import OmegaConf
from tqdm import tqdm


@dataclass
class GPUStatus:
    gpu_id: int
    occupied: bool


def setup_logger():
    Path("results/config_sweeper_logs/").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    now = datetime.datetime.now().__format__("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"results/config_sweeper_logs/{now}.log", "a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def custom_print(msg):
    logger.info(msg)
    try:
        tqdm.write(msg)
    except Exception:
        print(msg)


def gather_all_configs(config_dir):
    config_dir = Path(config_dir)

    all_configs = []
    for child in config_dir.glob("*"):
        if child.is_dir():
            all_configs += gather_all_configs(child)
        elif child.suffix == ".yaml":
            all_configs.append(child)
        else:
            logger.info(f"ignore: {child}")
    return all_configs


def check_exists(config_path):
    cfg = OmegaConf.load(str(config_path))
    exp_output_dir = cfg.runner_cfg.output_dir
    if Path(exp_output_dir).exists():
        msg = f"exp no.{current_job_id} exists: {exp_output_dir}"
        custom_print(msg)
        return True
    msg = f"exp no.{current_job_id} start: {exp_output_dir}"
    custom_print(msg)
    return False


def wait_until(process_queue: List, gpu_status: List[GPUStatus], sleep_time=10):
    # [TODO] event driven
    while True:
        return_flag = False
        for job_id, current_gpu_id, gpu_status_id, process in process_queue:
            if process.poll() is not None:
                returncode = process.poll()
                if returncode == 0:
                    msg = f"job {job_id}, finished!"
                else:
                    msg = f"job {job_id}, failed! return code: {returncode}"
                custom_print(msg)
                return_flag = True
                gpu_status[gpu_status_id].occupied = False
                process_queue.remove((job_id, current_gpu_id, gpu_status_id, process))
        if return_flag is True:
            return
        sleep(sleep_time)


def get_valid_gpu_id(gpu_status):
    for i, _gpu_status in enumerate(gpu_status):
        if _gpu_status.occupied is False:
            current_gpu_id = _gpu_status.gpu_id
            gpu_status_id = i
            _gpu_status.occupied = True
            return current_gpu_id, gpu_status_id
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-d", type=str)
    parser.add_argument("--max_num_gpus_used_in_parallel", type=int, default=8)
    parser.add_argument("--num_jobs_per_gpu", type=int, default=1)
    parser.add_argument("--start_gpu_id", type=int, default=0)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    all_configs = gather_all_configs(args.config_dir)
    msg = f"num of configs: {len(all_configs)}"
    custom_print(msg)

    start_gpu_id = args.start_gpu_id
    num_jobs_per_gpu = args.num_jobs_per_gpu
    max_num_gpus_used_in_parallel = args.max_num_gpus_used_in_parallel
    debug = args.debug
    max_jobs = max_num_gpus_used_in_parallel * num_jobs_per_gpu

    process_queue = list()
    gpu_status = [GPUStatus(gpu_id=i // num_jobs_per_gpu + start_gpu_id, occupied=False) for i in range(max_jobs)]
    gpu_status = gpu_status[::2] + gpu_status[1::2]
    for current_job_id, config_path in enumerate(tqdm(all_configs)):
        if len(process_queue) == max_jobs:
            msg = f"max_jobs {max_jobs} jobs, wait..."
            custom_print(msg)
            wait_until(process_queue, gpu_status)

        if check_exists(config_path):
            continue

        current_gpu_id, gpu_status_id = get_valid_gpu_id(gpu_status)
        cli = f"CUDA_VISIBLE_DEVICES={current_gpu_id} python scripts/run.py --config {str(config_path)} {'--test_only' if args.test_only else ''}"

        msg = f"CLI: {cli}"
        logger.info(msg)
        if not debug:
            process = subprocess.Popen(cli, shell=True)
        else:
            tqdm.write(msg)
            process = subprocess.Popen(f"sleep {current_gpu_id}".split())
        process_queue.append((current_job_id, current_gpu_id, gpu_status_id, process))

    while len(process_queue) != 0:
        wait_until(process_queue, gpu_status)
