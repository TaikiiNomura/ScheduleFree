from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Union, Tuple, Optional, Callable
from torch.optim.optimizer import ParamsT