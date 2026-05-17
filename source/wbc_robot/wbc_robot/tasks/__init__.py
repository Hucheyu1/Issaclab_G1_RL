# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##
from pathlib import Path

from isaaclab_tasks.utils import import_packages

REPLAY_DATASETS_DIR: str = str(Path(__file__).parent.parent.parent.parent.parent / "datasets" / "original_datasets")
EXTEMDED_DATASETS_DIR: str = str(Path(__file__).parent.parent.parent.parent.parent / "datasets" / "extend_datasets")

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
