# minimal_repro.py
from decord import VideoReader, cpu       # loads libgomp first
import torch         # then torch (with MKL) tries to init libiomp5 -> conflict
print("ok")