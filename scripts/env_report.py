"""Print the compute environment so model-size choices are explicit."""
import os
import platform

import torch

print(f"python      {platform.python_version()}")
print(f"torch       {torch.__version__}")
print(f"cuda        {torch.cuda.is_available()}")
print(f"cpu threads {os.cpu_count()} (torch uses {torch.get_num_threads()})")
try:
    import psutil
    print(f"ram GB      {psutil.virtual_memory().total / 1e9:.1f}")
except ImportError:
    pass
