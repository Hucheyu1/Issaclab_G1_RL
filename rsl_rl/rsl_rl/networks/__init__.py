# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural networks."""

from .memory import Memory
from .mlp import MLP
from .cnn import CNN

__all__ = [
    "CNN",
    "MLP",
    "Memory",
]
