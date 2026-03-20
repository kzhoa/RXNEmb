import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import qqtools as qt
import torch
from rdkit import RDLogger
from torch.nn.parallel import DistributedDataParallel as DDP


def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return qt.qDict(json.load(handle))


def setup_logger(log_dir):
    RDLogger.DisableLog("rdApp.*")
    RDLogger.DisableLog("rdApp.warning")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_path = log_dir / f"{datetime.now():%y%m%d-%H%Mh}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def param_norm(module):
    return math.sqrt(sum(param.norm().item() ** 2 for param in module.parameters()))


def grad_norm(module):
    return math.sqrt(sum(param.grad.norm().item() ** 2 for param in module.parameters() if param.grad is not None))


def get_lr(optimizer):
    return ",".join(str(round(group["lr"], 8)) for group in optimizer.param_groups)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model_state_dict(model):
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()
