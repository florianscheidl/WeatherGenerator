# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch
from astropy_healpix import healpy
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.model.engines import (
    EmbeddingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    Local2GlobalSumEngine,
    LocalAssimilationEngine,
    QueryAggregationEngine,
)

# from weathergen.model.model import ModelParams
from weathergen.model.parametrised_prob_dist import LatentInterpolator
from weathergen.model.positional_encoding import positional_encoding_harmonic


def _host_to_device_async(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Move a small host tensor to the device without stalling the compute stream.

    A pageable host-to-device copy synchronizes the stream; pinning first makes the
    copy truly asynchronous. On non-CUDA devices this is a plain copy.
    """
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


class EncoderModule(torch.nn.Module):
    name: "EncoderModule"

    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size) -> None:
        """
        Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: List of source sizes for each stream.
        :param stream_names: Ordered list of stream identifiers aligned with cf.streams.
        """
        super(EncoderModule, self).__init__()
        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.ae_aggregation_engine: QueryAggregationEngine | None = None
        self.ae_global_engine: GlobalAssimilationEngine | None = None
        self.ae_local_engine: LocalAssimilationEngine | None = None
        self.ae_local_global_engine: Local2GlobalAssimilationEngine | None = None
        self.embed_engine: EmbeddingEngine | None = None
        self.interpolator_latents: LatentInterpolator | None = None

        # embedding engine
        # determine stream names once so downstream components use consistent keys
        self.stream_names = list(cf.streams.keys())
        # separate embedding networks for differnt observation types
        self.embed_engine = EmbeddingEngine(cf, self.sources_size)

        assert cf.ae_global_att_dense_rate == 1.0, "Local attention not adapted for register tokens"
        self.num_register_tokens = cf.num_register_tokens
        self.num_class_tokens = cf.num_class_tokens

        # local assimilation engine
        self.ae_local_engine = LocalAssimilationEngine(cf)

        if cf.latent_noise_kl_weight > 0.0:
            self.interpolator_latents = LatentInterpolator(
                gamma=cf.latent_noise_gamma,
                dim=cf.ae_local_dim_embed,
                use_additive_noise=cf.latent_noise_use_additive_noise,
                deterministic=cf.latent_noise_deterministic_latents,
            )

        # local -> global assimilation engine adapter
        ae_adapter_type = cf.get("ae_adapter_type", "cross_attention")
        if ae_adapter_type == "sum":
            self.ae_local_global_engine = Local2GlobalSumEngine(cf)
        else:
            self.ae_local_global_engine = Local2GlobalAssimilationEngine(cf)

        # learnable queries
        if cf.ae_local_queries_per_cell:
            s = (self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
            # add meta data
            q_cells[:, :, -8:-6] = (
                (torch.arange(self.num_healpix_cells) / self.num_healpix_cells)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat((1, cf.ae_local_num_queries, 2))
            )
            theta, phi = healpy.pix2ang(
                nside=2**self.healpix_level, ipix=torch.arange(self.num_healpix_cells)
            )
            q_cells[:, :, -6:-3] = (
                torch.cos(theta).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -3:] = (
                torch.sin(phi).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -9] = torch.arange(cf.ae_local_num_queries)
            q_cells[:, :, -10] = torch.arange(cf.ae_local_num_queries)
        else:
            s = (1, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
        self.q_cells = torch.nn.Parameter(q_cells, requires_grad=True)

        # query aggregation engine
        self.ae_aggregation_engine = QueryAggregationEngine(cf, self.num_healpix_cells)

        # global assimilation engine
        self.ae_global_engine = GlobalAssimilationEngine(cf, self.num_healpix_cells)

    def forward(self, model_params, batch):
        """
        Encoder forward
        """

        stream_cell_tokens = checkpoint(
            self.embed_engine, batch, model_params.pe_embed, use_reentrant=False
        )

        tokens_global, posteriors = checkpoint(
            self.assimilate_local, model_params, stream_cell_tokens, batch, use_reentrant=False
        )

        tokens_global = checkpoint(
            self.ae_global_engine,
            tokens_global,
            coords=model_params.rope_coords,
            use_reentrant=False,
        )

        return tokens_global, posteriors

    def interpolate_latents(self, tokens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ "
        TODO
        """

        if self.cf.latent_noise_kl_weight > 0.0:
            tokens, posteriors = self.interpolator_latents.interpolate_with_noise(
                tokens, sampling=self.stage
            )
        else:
            posteriors = torch.zeros((1,), device=tokens.device)

        return tokens, posteriors

    def assimilate_local_project_chunked(
        self, tokens, tokens_global, cell_lens, q_cells_lens, cell_lens_cpu
    ):
        """
        Apply the local assimilation engine and then the
        local-to-global adapter using a chunking in the number of tokens
        to work around to bug in flash attention, the computations is performed in chunks

        cell_lens_cpu is the host copy of cell_lens; all chunk offsets, masks and index
        tensors are computed from it so that no device value is read on the host
        (slice bounds, boolean-mask indexing and empty-chunk checks would all force a
        host-device sync otherwise).
        """

        # subdivision factor for required splitting
        clen = self.num_healpix_cells // (2 if self.cf.healpix_level <= 5 else 8)
        tokens_global_unmasked = []
        posteriors = []

        # host-side token offsets per chunk
        cell_lens_csum_cpu = torch.cumsum(cell_lens_cpu, 0)
        zero_pad = torch.zeros(1, device=tokens.device, dtype=torch.int32)

        num_chunks = cell_lens_cpu.shape[0] // clen
        for i in range(num_chunks):
            # make sure we properly catch all elements in last chunk
            i_end = (i + 1) * clen if i < num_chunks - 1 else cell_lens_cpu.shape[0]
            l0 = 0 if i == 0 else int(cell_lens_csum_cpu[i * clen - 1])
            l1 = int(cell_lens_csum_cpu[i_end - 1])

            # if we have a very sparse input, we may have no tokens in the chunk;
            # skip processing of the empty chunk in this case
            if l0 == l1:
                continue

            toks = tokens[l0:l1]
            toks_global = tokens_global[i * clen : i_end]
            cell_lens_cur = torch.cat([zero_pad, cell_lens[i * clen : i_end]])

            # local assimilation model
            toks = self.ae_local_engine(toks, cell_lens_cur, use_reentrant=False)

            toks, posteriors_c = self.interpolate_latents(toks)
            posteriors += [posteriors_c]

            # select global tokens of unmasked cells via host-computed indices
            chunk_lens_cpu = cell_lens_cpu[i * clen : i_end]
            idxs_unmasked_cpu = torch.nonzero(chunk_lens_cpu, as_tuple=False).flatten()
            num_unmasked = idxs_unmasked_cpu.shape[0]
            idxs_unmasked = _host_to_device_async(idxs_unmasked_cpu, tokens.device)
            toks_global_unmasked = toks_global.index_select(0, idxs_unmasked)
            # q_cells_lens is [0, 1, 1, ...], so its first num_unmasked + 1 entries are
            # exactly the lens of the unmasked query sequences (incl. leading padding 0)
            q_cells_lens_unmasked = q_cells_lens[: num_unmasked + 1]
            cell_lens_unmasked = _host_to_device_async(
                torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32),
                        chunk_lens_cpu[idxs_unmasked_cpu].to(torch.int32),
                    ]
                ),
                tokens.device,
            )

            # local to global adapter engine
            toks_global_unmasked = self.ae_local_global_engine(
                toks,
                toks_global_unmasked,
                q_cells_lens_unmasked,
                cell_lens_unmasked,
            )

            tokens_global_unmasked += [toks_global_unmasked]

        if len(tokens_global_unmasked) == 0:
            assert False, "Not yet implemented"
        tokens_global_unmasked = torch.cat(tokens_global_unmasked)

        return tokens_global_unmasked, posteriors

    def aggregation_engine_unmasked(
        self,
        tokens_global_unmasked,
        tokens_global_register_class,
        tokens_lens_cpu,
        rope_cell_coords=None,
    ):
        """
        Aggregation engine on the global latents of unmasked cells

        tokens_lens_cpu is the host copy of the batch token lens; all length bookkeeping
        (split sizes, masks, sequence lens) runs on the host so that no device value is
        read back (each such read would force a host-device sync).
        """

        dev = tokens_global_unmasked.device

        # permute to use ae_local_num_queries as the batchsize and no_of_tokens
        # as seq len for flash attention
        tokens_global_unmasked = torch.permute(tokens_global_unmasked, [1, 0, 2])

        cell_lens_unflattened = torch.sum(tokens_lens_cpu, 2)
        cell_mask = cell_lens_unflattened.to(torch.bool)
        batch_lens = cell_mask.sum(dim=-1).flatten()
        expected_len = int(batch_lens.sum())
        actual_len = tokens_global_unmasked.shape[1]
        assert expected_len == actual_len, (
            f"Shape mismatch: expected {expected_len}, got {actual_len}"
        )
        tokens_global_unmasked = torch.split(tokens_global_unmasked.squeeze(0), batch_lens.tolist())
        tokens_global_unmasked = torch.cat(
            [
                t
                for tup in zip(tokens_global_register_class, tokens_global_unmasked, strict=False)
                for t in tup
            ],
            dim=0,
        )

        # Build packed coords matching the interleaved token order; cell indices of the
        # unmasked cells are computed on the host (boolean indexing on the device would
        # have a data-dependent output shape and thus sync)
        if rope_cell_coords is not None:
            num_extra = self.num_class_tokens + self.num_register_tokens
            zero_coords = torch.zeros(
                num_extra, 2, device=rope_cell_coords.device, dtype=rope_cell_coords.dtype
            )
            flat_mask = cell_mask.flatten(0, 1)
            # nonzero of the 2D mask is ordered by (sample, cell), matching the loop order
            cell_idxs_cpu = torch.nonzero(flat_mask, as_tuple=False)[:, 1].contiguous()
            cell_idxs = _host_to_device_async(cell_idxs_cpu, rope_cell_coords.device)
            coords_unmasked = rope_cell_coords.index_select(0, cell_idxs)
            packed_coords = []
            for coords_b in torch.split(coords_unmasked, flat_mask.sum(dim=-1).tolist()):
                packed_coords.append(zero_coords)
                packed_coords.append(coords_b)
            packed_coords = torch.cat(packed_coords, dim=0)
        else:
            packed_coords = None

        batch_lens = batch_lens + (self.num_class_tokens + self.num_register_tokens)
        batch_lens_patched = _host_to_device_async(
            torch.cat([torch.zeros(1, dtype=batch_lens.dtype), batch_lens], dim=0), dev
        )
        tokens_global_unmasked = self.ae_aggregation_engine(
            tokens_global_unmasked, batch_lens_patched, use_reentrant=False, coords=packed_coords
        )

        return tokens_global_unmasked

    def assimilate_local(
        self, model_params, tokens: torch.Tensor, batch: ModelBatch
    ) -> torch.Tensor:
        """
        Processes embedded tokens locally and prepares them for the global assimilation

        Args:
            model_params : Query and embedding parameters
            tokens : Input tokens to be processed by local assimilation
            cell_lens : Used to identify range of tokens to use from generated tokens in cell
                embedding
        Returns:
            Tokens for global assimilation
        """

        cell_lens = torch.sum(batch.tokens_lens, 2).flatten()

        # host copy of the lens for all host-side bookkeeping (chunk offsets, masks,
        # split sizes); falls back to a one-off device read if no host copy exists
        tokens_lens_cpu = getattr(batch, "tokens_lens_cpu", None)
        if tokens_lens_cpu is None:
            tokens_lens_cpu = batch.tokens_lens.cpu()
        cell_lens_cpu = torch.sum(tokens_lens_cpu, 2).flatten()

        num_steps_input = batch.get_num_source_steps()
        rs = num_steps_input * len(batch)

        # create register and latent tokens and prepend to latent spatial tokens
        num_extra_tokens = self.num_register_tokens + self.num_class_tokens
        pos_enc = positional_encoding_harmonic
        tokens_global_register_class = pos_enc(self.q_cells.repeat(rs, num_extra_tokens, 1))

        # TODO: re-enable or remove ae_local_queries_per_cell
        if self.cf.ae_local_queries_per_cell:
            tokens_global = (self.q_cells + model_params.pe_global).repeat(rs, 1, 1)
        else:
            num_tokens = self.num_healpix_cells
            tokens_global = self.q_cells.repeat(num_tokens, 1, 1) + model_params.pe_global
            tokens_global = tokens_global.repeat(rs, 1, 1)

        # apply local assimilation engine and project onto global latent vectors
        tokens_global_unmasked, posteriors = self.assimilate_local_project_chunked(
            tokens, tokens_global, cell_lens, model_params.q_cells_lens, cell_lens_cpu
        )

        # apply aggregation engine on unmasked tokens
        tokens_global_unmasked = self.aggregation_engine_unmasked(
            tokens_global_unmasked,
            tokens_global_register_class,
            tokens_lens_cpu,
            rope_cell_coords=model_params.rope_cell_coords,
        )

        # final processing

        tokens_global = (
            torch.permute(tokens_global, [1, 0, 2])
            .squeeze()
            .reshape(rs, self.num_healpix_cells, -1)
        )
        # TODO, TODO, TODO: do we need this
        tokens_global = torch.cat([tokens_global_register_class, tokens_global], dim=1)

        # fill positions of register/class tokens and unmasked cells with the aggregated
        # tokens; scatter indices are computed on the host since boolean-mask assignment
        # on the device would force a host-device sync (in forward and backward)
        num_extra = self.num_register_tokens + self.num_class_tokens
        mask_cpu = torch.cat(
            [
                torch.ones(rs, num_extra, dtype=torch.bool),
                cell_lens_cpu.reshape(rs, self.num_healpix_cells).to(torch.bool),
            ],
            dim=1,
        )
        scatter_idxs = _host_to_device_async(
            torch.nonzero(mask_cpu.flatten(), as_tuple=False).flatten(), tokens_global.device
        )
        tokens_global = (
            tokens_global.flatten(0, 1)
            .index_copy(0, scatter_idxs, tokens_global_unmasked.to(tokens_global.dtype))
            .reshape(tokens_global.shape)
        )

        # recover batch dimension and build global token list
        num_tokens_tot = self.num_healpix_cells + self.num_register_tokens + self.num_class_tokens
        q_c_shape = self.q_cells.shape
        tokens_global = (
            tokens_global.reshape([rs, num_tokens_tot, q_c_shape[-2], q_c_shape[-1]])
            #  removing this line because else they get added twice? + model_params.pe_global
        ).flatten(1, 2)

        return tokens_global, posteriors
