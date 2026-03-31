# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-teacher student-teacher model for cluster-based distillation."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class MultiTeacherStudentTeacher(nn.Module):
    """Student-teacher model with multiple teachers for cluster-based distillation.
    
    Each cluster has its own teacher model. During evaluation, the appropriate
    teacher is selected based on the cluster ID.
    """
    
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "MultiTeacherStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False

        mlp_input_dim_s = num_student_obs
        mlp_input_dim_t = num_teacher_obs
        self._teacher_hidden_dims = teacher_hidden_dims
        self._activation = activation
        self._num_actions = num_actions
        self._num_teacher_obs = num_teacher_obs

        # Student network
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        # Default teacher (used as template for creating new teachers)
        teacher_layers = []
        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        # Multi-teacher storage
        self.teachers: dict[int, nn.Sequential] = {}  # cluster_id -> teacher network
        self.teacher_normalizers: dict[int, EmpiricalNormalization] = {}  # cluster_id -> obs normalizer
        self.current_cluster_ids: torch.Tensor | None = None

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP template: {self.teacher}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def _create_teacher_network(self) -> nn.Sequential:
        """Create a new teacher network with the same architecture."""
        return copy.deepcopy(self.teacher)

    def set_cluster_ids(self, cluster_ids: torch.Tensor):
        """Set current cluster IDs for teacher selection."""
        self.current_cluster_ids = cluster_ids

    def load_teachers_from_directory(self, checkpoint_dir: str, cluster_range: tuple[int, int] | None = None, device="cpu"):
        """Load multiple teacher models from checkpoint directory.
        
        Args:
            checkpoint_dir: Path to directory containing cluster_* subdirectories
            cluster_range: Optional (min, max) range of clusters to load (inclusive)
            device: Device to load teachers onto
            
        Returns:
            dict: Mapping of cluster_id to normalizer state_dict (if available)
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise ValueError(f"Teacher checkpoint directory not found: {checkpoint_dir}")
        
        normalizer_state_dicts = {}
        
        # Find cluster directories
        cluster_dirs = []
        for item in checkpoint_path.iterdir():
            if item.is_dir():
                try:
                    cluster_id = int(item.name.split("_")[4])
                    if cluster_range is None or (cluster_range[0] <= cluster_id <= cluster_range[1]):
                        cluster_dirs.append((cluster_id, item))
                except (ValueError, IndexError):
                    continue
        
        if not cluster_dirs:
            raise ValueError(f"No valid cluster directories found in {checkpoint_dir}")
        
        cluster_dirs.sort(key=lambda x: x[0])
        print(f"[MultiTeacherStudentTeacher] Loading {len(cluster_dirs)} teachers...")
        
        for cluster_id, cluster_dir in cluster_dirs:
            norm_sd = self._load_single_teacher(cluster_id, cluster_dir, device)
            if norm_sd is not None:
                normalizer_state_dicts[cluster_id] = norm_sd
        
        # Create per-teacher normalizers from saved state dicts.
        # In single-teacher distillation the runner loads the RL actor's obs_norm_state_dict
        # into privileged_obs_normalizer so that teacher observations are normalised with
        # the same statistics the teacher was trained with.  For multi-teacher distillation
        # we must do this per teacher – each teacher was trained on a different data
        # distribution and therefore has different normalisation statistics.
        for cluster_id, norm_sd in normalizer_state_dicts.items():
            normalizer = EmpiricalNormalization(shape=[self._num_teacher_obs], until=1.0e8).to(device)
            normalizer.load_state_dict(norm_sd)
            normalizer.eval()  # freeze – we are not re-training the teachers
            self.teacher_normalizers[cluster_id] = normalizer
        
        if self.teachers:
            self.loaded_teacher = True
            print(f"[MultiTeacherStudentTeacher] Loaded teachers for clusters: {sorted(self.teachers.keys())}")
            print(f"[MultiTeacherStudentTeacher] Loaded normalizers for clusters: {sorted(self.teacher_normalizers.keys())}")
        
        return normalizer_state_dicts

    def _load_single_teacher(self, cluster_id: int, cluster_dir: Path, device="cpu") -> dict | None:
        """Load a single teacher from cluster directory.
        
        Returns:
            Normalizer state_dict if available, else None
        """
        model_files = list(cluster_dir.glob("model_*.pt"))
        if not model_files:
            print(f"[MultiTeacherStudentTeacher] Warning: No model files in {cluster_dir}")
            return None
        
        def get_iteration(path: Path) -> int:
            match = re.search(r"model_(\d+)\.pt", path.name)
            return int(match.group(1)) if match else 0
        
        latest_model = max(model_files, key=get_iteration)
        print(f"[MultiTeacherStudentTeacher] Loading cluster {cluster_id} from {latest_model.name}")
        
        checkpoint = torch.load(latest_model, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        
        # Extract actor parameters
        actor_state_dict = {}
        for key, value in state_dict.items():
            if "actor." in key:
                actor_state_dict[key.replace("actor.", "")] = value
        
        teacher = self._create_teacher_network()
        teacher.load_state_dict(actor_state_dict, strict=True)
        teacher.to(device)
        teacher.eval()
        self.teachers[cluster_id] = teacher
        
        return checkpoint.get("obs_norm_state_dict")

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.student(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.student(observations)

    def _normalize_for_teacher(self, teacher_observations: torch.Tensor, cluster_id: int) -> torch.Tensor:
        """Normalize observations using the per-teacher normalizer.

        In single-teacher distillation the runner's ``privileged_obs_normalizer``
        takes care of this.  For multi-teacher distillation each teacher was
        trained with its own normaliser, so we must apply it here instead.

        If no per-teacher normalizer is available (e.g. the teacher checkpoint
        did not contain ``obs_norm_state_dict``), the observations are returned
        unchanged – this keeps backward compatibility with setups that do not
        use empirical normalisation.
        """
        if cluster_id in self.teacher_normalizers:
            return self.teacher_normalizers[cluster_id](teacher_observations)
        return teacher_observations

    def evaluate(self, teacher_observations):
        """Get teacher actions based on current cluster IDs.

        When per-teacher normalizers are present (loaded via
        ``load_teachers_from_directory``), ``teacher_observations`` are expected
        to be **raw / unnormalized**.  Each teacher's observations are
        normalised with its own ``EmpiricalNormalization`` before inference,
        mirroring what ``privileged_obs_normalizer`` does in the single-teacher
        distillation path.
        """
        with torch.no_grad():
            if not self.teachers:
                return self.teacher(teacher_observations)
            
            if self.current_cluster_ids is None:
                # Use first available teacher
                default_id = min(self.teachers.keys())
                normed = self._normalize_for_teacher(teacher_observations, default_id)
                return self.teachers[default_id](normed)
            
            # Optimization for single teacher
            if len(self.teachers) == 1:
                teacher_id = list(self.teachers.keys())[0]
                normed = self._normalize_for_teacher(teacher_observations, teacher_id)
                return self.teachers[teacher_id](normed)
            
            # Multi-teacher: select & normalize based on cluster ID
            num_envs = teacher_observations.shape[0]
            device = teacher_observations.device
            actions = torch.zeros(num_envs, self._num_actions, device=device)
            
            for cluster_id in torch.unique(self.current_cluster_ids):
                cluster_id_int = cluster_id.item()
                mask = (self.current_cluster_ids == cluster_id)
                obs_subset = teacher_observations[mask]
                
                if cluster_id_int in self.teachers:
                    normed = self._normalize_for_teacher(obs_subset, cluster_id_int)
                    actions[mask] = self.teachers[cluster_id_int](normed)
                else:
                    # Fallback to nearest available teacher
                    nearest_id = self._find_nearest_teacher(cluster_id_int)
                    print(f"[WARNING] No teacher for cluster {cluster_id_int}, using nearest teacher {nearest_id} ({mask.sum().item()} envs)")
                    normed = self._normalize_for_teacher(obs_subset, nearest_id)
                    actions[mask] = self.teachers[nearest_id](normed)
            
            return actions

    def _find_nearest_teacher(self, cluster_id: int) -> int:
        """Find nearest available teacher for a cluster ID."""
        import bisect
        available_ids = sorted(self.teachers.keys())
        idx = bisect.bisect_left(available_ids, cluster_id)
        
        if idx == 0:
            return available_ids[0]
        elif idx == len(available_ids):
            return available_ids[-1]
        else:
            before, after = available_ids[idx - 1], available_ids[idx]
            return before if cluster_id - before <= after - cluster_id else after

    def load_state_dict(self, state_dict, strict=True):
        """Load model parameters.
        
        Returns:
            bool: True if resuming distillation training, False if loading from RL training
        """
        if any("actor" in key for key in state_dict.keys()):
            # Loading from RL training - load into default teacher
            teacher_state_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if "actor." in k}
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):
            # Resuming distillation training
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or actor parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
