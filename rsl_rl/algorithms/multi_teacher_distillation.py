# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-teacher distillation algorithm."""

import torch

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.modules import MultiTeacherStudentTeacher


class MultiTeacherDistillation(Distillation):
    """Multi-teacher distillation algorithm.
    
    Extends Distillation to support cluster-based teacher selection.
    The policy must be a MultiTeacherStudentTeacher.
    """

    policy: MultiTeacherStudentTeacher

    def set_cluster_ids(self, cluster_ids: torch.Tensor):
        """Set current cluster IDs for teacher selection.
        
        Args:
            cluster_ids: Tensor of shape (num_envs,) with cluster ID for each env
        """
        self.policy.set_cluster_ids(cluster_ids)
