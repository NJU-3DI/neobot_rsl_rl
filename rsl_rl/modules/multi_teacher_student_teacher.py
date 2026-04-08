# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-teacher student-teacher model for cluster-based distillation."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

from .actor_critic_transformer import ActorCriticTransformer


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
        student_class_name="MLP",
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.loaded_teacher = False
        self.student_class_name = student_class_name
        self._activation_name = activation

        mlp_input_dim_t = num_teacher_obs
        self._teacher_hidden_dims = teacher_hidden_dims
        self._num_actions = num_actions
        self._num_teacher_obs = num_teacher_obs

        # Student network
        self.student, unexpected_student_kwargs = self._build_student_network(
            num_student_obs=num_student_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            kwargs=kwargs,
        )
        self.is_recurrent = self.student.is_recurrent if hasattr(self.student, "is_recurrent") else False
        if unexpected_student_kwargs:
            print(
                "MultiTeacherStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str(unexpected_student_kwargs)
            )

        # Default teacher (used as template for creating new teachers)
        self.teacher = self._build_mlp(mlp_input_dim_t, teacher_hidden_dims, num_actions)
        self.teacher.eval()

        # Multi-teacher storage
        self.teachers: dict[int, nn.Sequential] = {}  # cluster_id -> teacher network
        self.teacher_normalizers: dict[int, EmpiricalNormalization] = {}  # cluster_id -> obs normalizer
        self.current_cluster_ids: torch.Tensor | None = None

        print(f"Student network ({self.student_class_name}): {self.student}")
        print(f"Teacher MLP template: {self.teacher}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) if not self.is_recurrent else None
        self.distribution = None
        Normal.set_default_validate_args = False

    def _build_mlp(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
        layers = [nn.Linear(input_dim, hidden_dims[0]), resolve_nn_activation(self._activation_name)]
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_index], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                layers.append(resolve_nn_activation(self._activation_name))
        return nn.Sequential(*layers)

    def _build_student_network(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        student_hidden_dims: list[int],
        activation: str,
        init_noise_std: float,
        noise_std_type: str,
        kwargs: dict,
    ) -> tuple[nn.Module, list[str]]:
        if self.student_class_name == "MLP":
            return self._build_mlp(num_student_obs, student_hidden_dims, num_actions), sorted(kwargs.keys())

        if self.student_class_name != "ActorCriticTransformer":
            raise ValueError(f"Unsupported student_class_name: {self.student_class_name}")

        student = ActorCriticTransformer(
            num_actor_obs=num_student_obs,
            num_critic_obs=num_teacher_obs,
            num_actions=num_actions,
            actor_hidden_dims=student_hidden_dims,
            critic_hidden_dims=self._teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        return student, []

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
        if self.is_recurrent:
            student_hidden_states = None if hidden_states is None else hidden_states[0]
            self.student.reset(dones=dones, hidden_states=student_hidden_states)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        if self.is_recurrent:
            return self.student.action_mean
        return self.distribution.mean

    @property
    def action_std(self):
        if self.is_recurrent:
            return self.student.action_std
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.is_recurrent:
            return self.student.entropy
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, masks=None, hidden_states=None):
        if self.is_recurrent:
            actions = self.student.act(observations, masks=masks, hidden_states=hidden_states)
            self.distribution = self.student.distribution
            return actions
        mean = self.student(observations)
        std = self.std.expand_as(mean)  # type: ignore[union-attr]
        self.distribution = Normal(mean, std)
        return None

    def act(self, observations, masks=None, hidden_states=None):
        if self.is_recurrent:
            actions = self.update_distribution(observations, masks=masks, hidden_states=hidden_states)
            return actions
        self.update_distribution(observations, masks=masks, hidden_states=hidden_states)
        return self.distribution.sample()

    def act_inference(self, observations, masks=None, hidden_states=None):
        if self.is_recurrent:
            return self.student.act_inference(observations, masks=masks, hidden_states=hidden_states)
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
        if self.is_recurrent:
            return self.student.get_hidden_states(), None
        return None

    def detach_hidden_states(self, dones=None):
        if self.is_recurrent:
            self.student.detach_hidden_states(dones)
