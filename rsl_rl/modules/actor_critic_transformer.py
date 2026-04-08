# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl.networks import RGMTMemory, TransformerMemory

from .actor_critic import ActorCritic


class ActorCriticTransformer(ActorCritic):
    """基于 Transformer 时序记忆的 Actor-Critic 架构。

    支持两种模式：
    - **默认（单流）**：观测序列经因果 Transformer (TransformerMemory) 编码后送入 Actor/Critic。
    - **RGMT（双流）**：历史编码器 + 命令编码器（见论文 Robust and Generalized Humanoid Motion Tracking），
      输出 dynamics-conditioned 的命令嵌入 u_t，与当前观测拼接后送入 Actor/Critic。

    RGMT 模式下观测需为 [proprio, command] 拼接，且 num_proprio_obs + num_command_obs = num_actor_obs。
    """

    is_recurrent = True  # 与 recurrent 模型共用 rollout 存储路径

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        transformer_hidden_dim=128,
        transformer_num_layers=2,
        transformer_num_heads=4,
        transformer_context_len=64,
        transformer_dropout=0.0,
        use_rgmt_architecture=False,
        num_proprio_obs=None,
        num_command_obs=None,
        history_len=10,
        command_half_len=10,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ 收到未预期参数，将忽略: "
                + str([key for key in kwargs.keys()])
            )

        self.use_rgmt = use_rgmt_architecture
        # 输入检查
        if self.use_rgmt:
            if num_proprio_obs is None or num_command_obs is None:
                raise ValueError("RGMT 模式需要 num_proprio_obs 和 num_command_obs")
            if num_proprio_obs + num_command_obs != num_actor_obs:
                raise ValueError(
                    f"num_proprio_obs ({num_proprio_obs}) + num_command_obs ({num_command_obs}) "
                    f"必须等于 num_actor_obs ({num_actor_obs})"
                )
            self.num_proprio_obs = num_proprio_obs
            self.num_command_obs = num_command_obs
            # Critic 输入 = s_t = [o_t, g_t, o_t^priv]
            actor_mlp_dim = num_proprio_obs + transformer_hidden_dim
            critic_mlp_dim = num_critic_obs
        else:
            actor_mlp_dim = transformer_hidden_dim
            critic_mlp_dim = transformer_hidden_dim

        super().__init__(
            num_actor_obs=actor_mlp_dim,
            num_critic_obs=critic_mlp_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        if self.use_rgmt:
            self.memory_a = RGMTMemory(
                num_proprio_obs=num_proprio_obs,
                num_command_obs=num_command_obs,
                hidden_size=transformer_hidden_dim,
                history_len=history_len,
                command_half_len=command_half_len,
                num_heads=transformer_num_heads,
                dropout=transformer_dropout,
            )
            self.memory_c = None
            self._last_u_t = None
        else:
            self.memory_a = TransformerMemory(
                input_size=num_actor_obs,
                hidden_size=transformer_hidden_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                context_len=transformer_context_len,
                dropout=transformer_dropout,
            )
            self.memory_c = TransformerMemory(
                input_size=num_critic_obs,
                hidden_size=transformer_hidden_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                context_len=transformer_context_len,
                dropout=transformer_dropout,
            )
            self._last_u_t = None

    def reset(self, dones=None, hidden_states=None):
        actor_hidden_states = None
        critic_hidden_states = None
        if hidden_states is not None:
            actor_hidden_states = hidden_states[0]
            if len(hidden_states) > 1:
                critic_hidden_states = hidden_states[1]
        self.memory_a.reset(dones, actor_hidden_states)
        if self.memory_c is not None:
            self.memory_c.reset(dones, critic_hidden_states)

    def _get_actor_input(self, observations, u_t):
        if self.use_rgmt:
            proprio = observations[..., : self.num_proprio_obs]
            return torch.cat([proprio, u_t], dim=-1)
        return u_t

    def _get_critic_input(self, critic_observations, u_t):
        return critic_observations

    def act(self, observations, masks=None, hidden_states=None):
        u_t = self.memory_a(observations, masks, hidden_states)
        self._last_u_t = u_t
        if u_t.dim() == 3 and u_t.size(0) == 1:
            u_t = u_t.squeeze(0)
        input_a = self._get_actor_input(observations, u_t)
        return super().act(input_a)

    def act_inference(self, observations, masks=None, hidden_states=None):
        actor_hidden_states = hidden_states[0] if hidden_states is not None else None
        u_t = self.memory_a(observations, masks, actor_hidden_states)
        if u_t.dim() == 3:
            u_t = u_t.squeeze(0)
        input_a = self._get_actor_input(observations, u_t)
        return super().act_inference(input_a)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        if self.use_rgmt:
            input_c = self._get_critic_input(critic_observations, None)
        else:
            input_c = self.memory_c(critic_observations, masks, hidden_states)
            if input_c.dim() == 3:
                input_c = input_c.squeeze(0) if input_c.size(0) == 1 else input_c
        return super().evaluate(input_c)

    def get_hidden_states(self):
        if self.use_rgmt:
            hs = self.memory_a.hidden_states
            if hs[0] is None or hs[1] is None:
                return (None, None)  # Storage will skip saving
            return hs, hs
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def detach_hidden_states(self, dones=None):
        self.memory_a.detach_hidden_states(dones)
        if self.memory_c is not None:
            self.memory_c.detach_hidden_states(dones)
