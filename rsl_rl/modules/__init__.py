# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_transformer import ActorCriticTransformer
from .multi_teacher_student_teacher import MultiTeacherStudentTeacher
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "ActorCriticTransformer",
    "EmpiricalNormalization",
    "MultiTeacherStudentTeacher",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "StudentTeacherRecurrent",
]
