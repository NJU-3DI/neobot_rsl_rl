from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维：sin
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])  # 奇数维：cos
        pe = pe.unsqueeze(1)  
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)] 
        return self.dropout(x)

class TransformerMemory(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        context_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_len = context_len

        self.input_proj = nn.Linear(input_size, hidden_size)

        # Pos-encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=context_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # 使用 (seq, batch, feature) 约定
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 推理时的上下文缓冲区，形状 (context_len, num_envs/batch size, hidden_size)
        self.context_buffer: torch.Tensor | None = None

    def forward(self, input: torch.Tensor, masks: torch.Tensor | None = None, hidden_states=None) -> torch.Tensor:
        batch_mode = masks is not None
        if batch_mode:
            return self._forward_batch(input, masks)
        else:
            return self._forward_inference(input)

    def _forward_batch(self, input: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """训练前向：整段序列一次处理。

        操作步骤：
        1. 输入投影：将 (seq_len, batch, input_size) 映射到 hidden_size
        2. 位置编码：加在输入上
        3. 因果 mask：保证第 τ 步只能看到 ≤ τ 的 token（自回归）
        4. padding mask：标记填充位置，在 attention 中忽略
        5. Transformer 编码：输出 (seq_len, batch, hidden_size)
        """
        seq_len = input.size(0)

        x = self.input_proj(input)  # 线性投影到隐藏维
        x = self.pos_encoder(x)     # 加位置编码

        # 因果 mask：上三角为 -inf，使注意力只看过去
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input.device)

        # padding mask：masks 中 True 表示有效，需取反得到「需屏蔽的填充位置」
        key_padding_mask = ~masks.permute(1, 0).bool()  # (batch, seq_len)

        out = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return out  # (seq_len, batch, hidden_size)

    def _forward_inference(self, input: torch.Tensor) -> torch.Tensor:
        """推理前向：单步输入，用滑动窗口缓存历史。

        操作步骤：
        1. 将新观测投影后追加到 context_buffer
        2. 若超出 context_len，截断只保留最近的一段
        3. 对整段缓冲做位置编码 + 因果 Transformer
        4. 只返回最后一个时间步的输出（当前步的表示）
        """
        batch_size = input.size(0)
        x = self.input_proj(input).unsqueeze(0)  # (1, batch, hidden_size)

        if self.context_buffer is None or self.context_buffer.size(1) != batch_size:
            self.context_buffer = x
        else:
            self.context_buffer = torch.cat([self.context_buffer, x], dim=0)
            if self.context_buffer.size(0) > self.context_len:
                self.context_buffer = self.context_buffer[-self.context_len :]  # 滑动窗口截断

        ctx = self.pos_encoder(self.context_buffer)
        ctx_len = ctx.size(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(ctx_len, device=input.device)

        out = self.transformer_encoder(ctx, mask=causal_mask)
        return out[-1:, :, :]  # 只取最后一步 (1, batch, hidden_size)

    def reset(self, dones: torch.Tensor | None = None, hidden_states=None):
        """重置缓冲区：dones 为 None 时清空全部，否则只清零已结束环境的对应位置。"""
        if dones is None:
            self.context_buffer = hidden_states
        elif self.context_buffer is not None:
            done_mask = (dones == 1).view(1, -1, 1)
            self.context_buffer = torch.where(done_mask, torch.zeros_like(self.context_buffer), self.context_buffer)

    def detach_hidden_states(self, dones: torch.Tensor | None = None):
        """将缓冲区从计算图分离，用于截断梯度。"""
        if self.context_buffer is not None:
            if dones is None:
                self.context_buffer = self.context_buffer.detach()
            else:
                done_mask = (dones == 1).view(1, -1, 1)
                detached = self.context_buffer.detach()
                self.context_buffer = torch.where(done_mask, detached, self.context_buffer)

    @property
    def hidden_states(self):
        """返回上下文缓冲区，供 rollout storage 保存。"""
        return self.context_buffer


class RGMTMemory(nn.Module):
    """RGMT 双流记忆模块：基于 dynamics-conditioned 的命令聚合（见论文 Robust and Generalized Humanoid Motion Tracking）。

    结构：
    - 历史编码器：本体历史 o_{t-K:t} → MLP → 位置编码 → 1 层因果 Transformer → 时间维 MaxPool → h_t
    - 命令编码器：以 h_t 为 query，对命令窗口 g_{t-L:t+L} 做 cross-attention → u_t
    - 输出：u_t，供 Actor 与当前观测拼接使用
    """

    def __init__(
        self,
        num_proprio_obs: int,
        num_command_obs: int,
        hidden_size: int = 128,
        history_len: int = 10,
        command_half_len: int = 10,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_proprio_obs = num_proprio_obs
        self.num_command_obs = num_command_obs
        self.hidden_size = hidden_size
        self.history_len = history_len
        self.command_half_len = command_half_len
        self.command_window_len = 2 * command_half_len + 1

        # History encoder: 2 layer MLP, map proprio to hidden
        self.proprio_mlp = nn.Sequential(
            nn.Linear(num_proprio_obs, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.history_pos_enc = PositionalEncoding(hidden_size, max_len=history_len, dropout=dropout)

        # 1 layer causal Transformer, model history sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,  # Pre-LN：先 LayerNorm 再注意力
        )
        self.history_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.history_output_norm = nn.LayerNorm(hidden_size)  

        # Command encoder: MLP map command to hidden
        self.command_mlp = nn.Sequential(
            nn.Linear(num_command_obs, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.command_pos_enc = PositionalEncoding(
            hidden_size, max_len=self.command_window_len, dropout=dropout
        )

        # Query 投影：将 h_t 投影为 cross-attention 的 query
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Cross-Attention 块（Pre-LN）：MHA + FFN，残差连接
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.cross_attn_norm1 = nn.LayerNorm(hidden_size)
        self.cross_attn_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.cross_attn_norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn_norm3 = nn.LayerNorm(hidden_size)

        # 推理时的缓冲：本体历史、命令窗口
        self.proprio_buffer: torch.Tensor | None = None
        self.command_buffer: torch.Tensor | None = None

    def _history_encoder_forward(self, proprio: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        """历史编码器：本体历史 → 动力学嵌入 h_t。

        操作步骤：
        1. MLP 投影本体观测到 hidden 维
        2. 加位置编码
        3. 因果 Transformer 编码
        4. 时间维 MaxPool：对每个维度取最大，得到紧凑的 h_t
        5. 用 mask 屏蔽填充位置，避免 -inf 进入结果
        """
        seq_len = proprio.size(0)
        x = self.proprio_mlp(proprio)
        x = self.history_pos_enc(x)
        # Et−K:t = MLP(ot−K:t) ∈ R(K+1)×nembd, E ̃ t−K:t = Et−K:t + P,

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=proprio.device)
        key_padding_mask = None
        if masks is not None:
            key_padding_mask = ~masks.permute(1, 0).bool()

        out = self.history_transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        # H(1) = H(0) + MHA(LN(H(0))), H(2) = H(1) + MLP(LN(H(1)))
        out = self.history_output_norm(out)  
        # H̄ = LN(H^(2)) 

        if masks is not None:
            out = out.masked_fill(~masks.permute(1, 0).unsqueeze(-1), float("-inf"))
        h_t = out.max(dim=0).values
        # ht[j] = max  τ ∈{t−K,...,t}  H ̄ τ [j]
        
        if masks is not None:
            h_t = torch.where(torch.isfinite(h_t), h_t, torch.zeros_like(h_t))  # 无效位置填 0
        return h_t  # (batch, hidden_size)

    def _command_encoder_forward(
        self, h_t: torch.Tensor, command: torch.Tensor, masks: torch.Tensor | None
    ) -> torch.Tensor:

        q = self.query_proj(h_t).unsqueeze(0)  # (1, batch, hidden)
        # q_t = MLP_dyn(h_t)
        q_norm = self.cross_attn_norm1(q)
        # q_norm = LN(q_t)

        z = self.command_mlp(command)
        z = self.command_pos_enc(z)
        # Z̃ = MLP_cmd(g_{t-L:t+L}) + P^{cmd}

        key_padding_mask = None
        if masks is not None:
            key_padding_mask = ~masks.permute(1, 0).bool()
        attn_out, _ = self.cross_attn(
            query=q_norm, key=z, value=z, key_padding_mask=key_padding_mask
        )
        s = q + attn_out   
        # s^(1) = q_t + MHA(LN(q_t), Z̃)

        s = s + self.cross_attn_ffn(self.cross_attn_norm2(s))  # FFN + 残差
        # s^(2) = s^(1) + MLP(LN(s^(1)))

        u_t = self.cross_attn_norm3(s)
        # u_t = LN(s^(2))
        return u_t.squeeze(0)  

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        """前向：有 masks 为训练（批量序列），否则为推理（单步）。输入为 [proprio, command] 拼接。"""
        if masks is not None:
            return self._forward_batch(input, masks)
        return self._forward_inference(input)

    def _forward_batch(self, input: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """训练前向：逐时间步处理，每步取历史 o_{t-K:t} 和命令窗口 g_{t-L:t+L}（含未来，训练时有完整轨迹）。"""
        seq_len, batch_size, _ = input.shape
        proprio = input[..., : self.num_proprio_obs]
        command = input[..., self.num_proprio_obs :]

        K = self.history_len - 1
        L = self.command_half_len
        outputs = []
        for t in range(seq_len):
            # 历史窗口 o_{t-K:t}
            start_h = max(0, t - K)
            end_h = t + 1
            p_seq = proprio[start_h:end_h]
            m_h = masks[start_h:end_h] if masks is not None else None
            h_t = self._history_encoder_forward(p_seq, m_h)

            # 命令窗口 g_{t-L:t+L}（含未来），超出序列边界时用首/尾帧填充
            start_c = max(0, t - L)
            end_c = min(seq_len, t + L + 1)
            c_seq = command[start_c:end_c]
            pre_pad = max(0, L - t)   # 前面不足时用首帧填充
            post_pad = max(0, (t + L + 1) - seq_len)  # 后面不足时用尾帧填充
            if pre_pad > 0:
                c_seq = torch.cat([command[0:1].expand(pre_pad, -1, -1), c_seq], dim=0)
            if post_pad > 0:
                c_seq = torch.cat([c_seq, command[seq_len - 1 : seq_len].expand(post_pad, -1, -1)], dim=0)
            c_seq = c_seq[: self.command_window_len]  # 截断到固定长度

            m_c = torch.ones(
                self.command_window_len, batch_size, dtype=torch.bool, device=input.device
            )
            u_t = self._command_encoder_forward(h_t, c_seq, m_c)
            outputs.append(u_t)
        out = torch.stack(outputs, dim=0)
        return out  # (seq_len, batch, hidden_size)

    def _forward_inference(self, input: torch.Tensor) -> torch.Tensor:
        """推理前向：单步输入。流式推理无未来命令，使用 [g_{t-2L}, ..., g_t]，不足时用 g_t 填充。"""
        batch_size = input.size(0)
        proprio = input[:, : self.num_proprio_obs]  # 切分本体
        command = input[:, self.num_proprio_obs :]  # 切分命令

        # 追加到缓冲区
        p_new = proprio.unsqueeze(0)
        c_new = command.unsqueeze(0)
        if self.proprio_buffer is None or self.proprio_buffer.size(1) != batch_size:
            self.proprio_buffer = p_new
            self.command_buffer = c_new
        else:
            self.proprio_buffer = torch.cat([self.proprio_buffer, p_new], dim=0)
            self.command_buffer = torch.cat([self.command_buffer, c_new], dim=0)
            if self.proprio_buffer.size(0) > self.history_len:
                self.proprio_buffer = self.proprio_buffer[-self.history_len :]  # 滑动窗口截断
            if self.command_buffer.size(0) > self.command_window_len:
                self.command_buffer = self.command_buffer[-self.command_window_len :]

        # 命令窗口不足时用最后一个 g_t 向前填充
        p_seq = self.proprio_buffer
        c_seq = self.command_buffer
        if c_seq.size(0) < self.command_window_len:
            pad_len = self.command_window_len - c_seq.size(0)
            c_seq = torch.cat([c_seq[-1:].expand(pad_len, -1, -1), c_seq], dim=0)
        c_seq = c_seq[-self.command_window_len :]

        h_t = self._history_encoder_forward(p_seq, None)
        u_t = self._command_encoder_forward(h_t, c_seq, None)
        return u_t.unsqueeze(0)  # (1, batch, hidden)

    def reset(self, dones: torch.Tensor | None = None, hidden_states=None):
        """重置缓冲：dones 为 None 时清空全部，否则只清零已结束环境的对应位置。"""
        if dones is None:
            if hidden_states is None:
                self.proprio_buffer = None
                self.command_buffer = None
            else:
                self.proprio_buffer, self.command_buffer = hidden_states
        elif self.proprio_buffer is not None:
            done_mask = (dones == 1).view(1, -1, 1)
            self.proprio_buffer = torch.where(done_mask, torch.zeros_like(self.proprio_buffer), self.proprio_buffer)
            self.command_buffer = torch.where(done_mask, torch.zeros_like(self.command_buffer), self.command_buffer)

    def detach_hidden_states(self, dones: torch.Tensor | None = None):
        """将缓冲从计算图分离，用于截断梯度。"""
        if self.proprio_buffer is not None:
            if dones is None:
                self.proprio_buffer = self.proprio_buffer.detach()
                self.command_buffer = self.command_buffer.detach()
            else:
                done_mask = (dones == 1).view(1, -1, 1)
                detached_proprio = self.proprio_buffer.detach()
                detached_command = self.command_buffer.detach()
                self.proprio_buffer = torch.where(done_mask, detached_proprio, self.proprio_buffer)
                self.command_buffer = torch.where(done_mask, detached_command, self.command_buffer)

    @property
    def hidden_states(self):
        """返回 (本体缓冲, 命令缓冲)，供 rollout storage 保存。"""
        return (self.proprio_buffer, self.command_buffer)
