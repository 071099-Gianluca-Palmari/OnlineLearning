from operator import mul
import numpy as np
import torch
import torch.nn as nn
import pandas
import glob
import yaml
import subprocess
import os


def save_git_hash(cwd):
    git_hash = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"],
        cwd=cwd)
    git_hash = git_hash.decode("utf-8")
    with open('git_hash.txt', 'w') as f:
        f.write(git_hash)