# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .multi_teacher_distillation import MultiTeacherDistillation
from .ppo import PPO

__all__ = ["PPO", "Distillation", "MultiTeacherDistillation"]
