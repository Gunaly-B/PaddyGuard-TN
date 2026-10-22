import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy, and torch for CPU reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Configure CPU threading according to hardware constraint
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    print(f"[Seed & Threads] Fixed seed={seed}, torch threads={torch.get_num_threads()} (CPU count={cpu_count})")

if __name__ == "__main__":
    set_seed(42)
