import os
import sys

with open(sys.argv[0], "r") as f:
    code = f.read()


import random
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch


@dataclass
class Config:
    seed: int = 11
    experiments_dir: str = "workdir/experiments"


c = Config()

random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)

assert torch.cuda.is_available()

device = "cuda"

ts = datetime.now().strftime("%y%m%d-%H%M%S")
uid = uuid.uuid4().hex[:8]
e_id = f"{ts}-{uid}"
e_dir = os.path.join(c.experiments_dir, e_id)
os.makedirs(e_dir, exist_ok=True)
log_path = os.path.join(e_dir, f"{e_id}-log.txt")


def print0(s, console=False):
    with open(log_path, "a") as f:
        if console:
            print(s)
        print(s, file=f)


print0(code)
print0("=" * 100)


class TabICLv2:
    def __init__(self, config=None):
        pass


print0("=" * 100)
print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)
print0(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"experiment done: {e_id}", console=True)
