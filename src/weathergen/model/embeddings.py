# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
import torch.distributed.nn  # autograd-aware all_gather; not pulled in by `import torch`
from torch.utils.checkpoint import checkpoint

from weathergen.model.attention import MultiSelfAttentionHead
from weathergen.model.layers import MLP

# from weathergen.model.mlp import MLP
from weathergen.model.norms import RMSNorm
from weathergen.model.positional_encoding import positional_encoding_harmonic


class StreamEmbedTransformer(torch.nn.Module):
    def __init__(
        self,
        num_tokens,
        token_size,
        num_channels,
        dim_embed,
        dim_out,
        num_blocks,
        num_heads,
        dropout_rate=0.0,
        norm_type="LayerNorm",
        unembed_mode="full",
        stream_name="stream_embed",
    ):
        """Constructor

        unembed_mode : { 'full' , 'block'}
          full : monolithic (and correspondingly large) unembedding network that maps from
                 (num_tokens x dim_embed) to dim_out, allowing for mixing between channels/columns
          block : per-channel/column unembedding network
                (which is hence a block-sparse form of full)
        """

        super(StreamEmbedTransformer, self).__init__()

        self.name = f"StreamEmbedder_{stream_name}"
        self.num_tokens = num_tokens
        self.token_size = token_size
        self.num_channels = num_channels
        self.dim_in = token_size
        self.dim_embed = dim_embed
        self.dim_out = dim_out
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.unembed_mode = unembed_mode

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_blocks):
            self.layers.append(
                MultiSelfAttentionHead(
                    self.dim_embed,
                    self.num_heads,
                    dropout_rate=dropout_rate,
                    with_qk_lnorm=True,
                    with_flash=True,
                    # without this the blocks silently keep the default LayerNorm, so
                    # norm_type reached only ln_final and never the attention prologue
                    norm_type=norm_type,
                )
            )
            self.layers.append(
                MLP(
                    self.dim_embed,
                    self.dim_embed,
                    hidden_factor=2,
                    dropout_rate=dropout_rate,
                    with_residual=True,
                    norm_type=norm_type,
                )
            )

        self.embed = torch.nn.Linear(self.dim_in, self.dim_embed)

        if self.unembed_mode == "full":
            self.ln_final = norm(num_channels * self.dim_embed, eps=1e-03)
            self.unembed = torch.nn.Linear(
                num_channels * self.dim_embed,
                self.num_tokens * self.dim_out,
            )

        elif self.unembed_mode == "block":
            dim_out = (self.num_tokens * self.dim_out) // num_channels
            self.unembed = torch.nn.ModuleList(
                [torch.nn.Linear(dim_embed, dim_out) for _ in range(num_channels)]
            )
            self.ln_final = torch.nn.ModuleList(
                [norm(dim_embed, eps=1e-6) for _ in range(num_channels)]
            )
        else:
            raise ValueError(f"Unknown unembed mode: {unembed_mode}")

        self.dropout_final = torch.nn.Dropout(0.1)

    def forward(self, x_in):
        peh = positional_encoding_harmonic

        # embed provided input data
        # NOTE: no checkpoint on self.embed, a single Linear whose input is x_in (alive
        # regardless) and whose output is retained as the first block's checkpoint input
        x = peh(self.embed(x_in.transpose(-2, -1)))

        # x is replicated across ranks, so each rank runs the blocks on one shard of dim 0
        # (a batch dimension the blocks never mix across) and the shards are gathered back
        # afterwards. NCCL's all-gather requires equal-sized contributions, so the shards
        # are uniform and the last one is zero-padded; the padding is trimmed after the
        # gather. Without a process group this degenerates to world_size 1, i.e. no split.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            local_rank = torch.distributed.get_rank()
        else:
            world_size, local_rank = 1, 0

        num_rows = x.shape[0]
        shard_rows = (num_rows + world_size - 1) // world_size
        x_slice = x[local_rank * shard_rows : (local_rank + 1) * shard_rows]
        pad_rows = shard_rows - x_slice.shape[0]
        if pad_rows > 0:
            x_slice = torch.nn.functional.pad(x_slice, [0, 0] * (x.ndim - 1) + [0, pad_rows])

        # One checkpoint per layer, not one per (attention, MLP) pair. Pairing them retains
        # one boundary activation per block instead of two, but the recompute then has to
        # rebuild a whole block's graph before the backward consumes any of it, and every
        # intermediate is live at once. Measured on run fy0tazen: that burst was 14.7 GiB
        # over the surrounding plateau and set the run's peak, against ~1.2 GiB of boundary
        # activations saved by pairing. The finer split trades the cheaper quantity for the
        # expensive one; it does not change how much is recomputed.
        for layer in self.layers:
            x_slice = checkpoint(layer, x_slice, use_reentrant=False)

        if world_size > 1:
            # autograd-aware all-gather: the plain torch.distributed one is an in-place
            # collective with no backward, which would strand the blocks without gradients
            x = torch.cat(torch.distributed.nn.functional.all_gather(x_slice), dim=0)
            x = x[:num_rows]
        else:
            x = x_slice

        return checkpoint(self.readout, x, use_reentrant=False)

    def readout(self, x):
        """
        Map the per-channel embeddings onto the output tokens.

        Checkpointed by forward: in 'full' mode ln_final alone materialises a tensor as
        large as x, which is otherwise held until the backward pass.
        """

        if self.unembed_mode == "full":
            out = self.unembed(self.ln_final(x.flatten(-2, -1)))
        elif self.unembed_mode == "block":
            out = [
                ue(ln(x[:, i]))
                for i, (ue, ln) in enumerate(zip(self.unembed, self.ln_final, strict=True))
            ]
            out = torch.stack(out, dim=1).flatten(-2, -1)
        else:
            raise ValueError(f"Unknown unembed mode: {self.unembed_mode}")

        if out.shape[-1] < self.dim_out:
            out = torch.nn.functional.pad(out, [0, self.dim_out - out.shape[-1]], value=0.0)
        # final reshape
        out = self.dropout_final(out.reshape(-1, self.num_tokens, self.dim_out))

        return out


class StreamEmbedLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, stream_name="stream_embed"):
        """Constructor"""

        super(StreamEmbedLinear, self).__init__()

        self.name = f"StreamEmbedder_{stream_name}"
        self.layer = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # NOTE: no checkpoint, a single Linear whose input is the argument itself
        x = self.layer(x.flatten(-2, -1)).unsqueeze(0)

        return x
