<p align="center">
  <img src="assets/weathergenerator_logo.png" alt="WeatherGenerator" width="400px">
</p>

<div align="center">
  <h1>The WeatherGenerator <br> Machine Learning Earth System Model</h1>
</div>

The WeatherGenerator project is developing a machine learning-based Earth system model. 
It will be trained on a wide range of datasets, including reanalyses, forecast data and observations, to provide a robust and versatile model for the dynamics.
Through this, it can be used for a wide-range of applications. General updates are shared on the project website: [weathergenerator.eu](https://weathergenerator.eu/) 

More details coming soon. Please open an issue if you are interested in using the model.

<hr>

<p align="center">
  <img src="assets/weathergenerator_partner.png" alt="Partners" width="1000px">
</p>

# How to use the WeatherGenerator project

The model is currently being developed by the WeatherGenerator Consortium. If you want to
engage, you are encouraged to contact us first by opening an issue on Github.

# Development guidelines

The [main branch](https://github.com/ecmwf/WeatherGenerator/tree/main) is the most stable version. If you are running experiments, you should use this branch.

The [develop branch](https://github.com/ecmwf/WeatherGenerator/tree/develop) has the latest
features. However, it is currently evolving at a fast pace. It should not be expected to have stable code or weight interfaces, or to be backward compatible.

# Copyright and License

This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.


---

## Running WeatherGenerator on your machine

This setup is WIP. It only works for the single dataset configuration referenced below.

### Supported hardware

| Hardware | torch extra | Attention | Status |
|---|---|---|---|
| NVIDIA GPU (CUDA) | `gpu` | flash-attn | supported |
| AMD GPU (ROCm 6.4) | `rocm` | flash-attn if built, else torch SDPA | untested on AMD hardware |
| Apple Silicon (MPS) | `cpu` | torch SDPA | tested |
| Intel GPU (XPU) | `xpu` | torch SDPA | implemented, untested |
| CPU | `cpu` | torch SDPA | tested |

Without flash-attn, attention falls back to torch's `scaled_dot_product_attention`. It pads
the packed sequences, so it needs more memory than flash-attn and is meant for small runs.

### Installation and setup

1. Install uv, see https://docs.astral.sh/uv/getting-started/installation/.
2. Clone the repo and cd to `WeatherGenerator`
3. Create output directories and run sync script:
    ```bash
    mkdir -p logs models output results
    ./scripts/actions.sh sync
    ```
    `sync` picks the torch extra for the machine: `cpu` on macOS (includes MPS), `rocm` or `xpu`
    if an AMD or Intel GPU is detected, otherwise `gpu`. Override it with e.g.
    `WEATHERGEN_TORCH_EXTRA=cpu ./scripts/actions.sh sync` on a Linux machine without GPU.

**AMD GPUs:** there is no prebuilt flash-attn for ROCm. Either build it (see
[flash-attention](https://github.com/Dao-AILab/flash-attention), e.g. with its Triton backend
via `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), or run with `--options with_flash_attention=False`
to use torch's attention.

###  Download data
- ERA5, 2020, 1-month.
    ```bash
    uv run --with "anemoi-datasets[remote]" anemoi-datasets create --overwrite datasets/download_configs/era5_o96_2020_1m.yaml datasets/era5-o96-2020-1pct-6h-v1.zarr
    ```


### Training

```bash
WEATHERGEN_PRIVATE_CONF=./private_config.yaml uv run train --base-config ./config/era5_local.yml
```

The device is chosen automatically (`device: "auto"`: the available GPU, else the CPU). Options,
set with `--options key=value` or in a config file:

- `device=cpu` (or `cuda`, `mps`, `xpu`; `cuda` also covers ROCm): use a specific device, e.g. the
  CPU on a machine with a GPU.
- `with_mixed_precision=False attention_dtype=fp32`: train in fp32 instead of bf16 mixed
  precision. On CPUs this is much faster, since most CPUs have no fast bf16.

For example, training on the CPU:

```bash
WEATHERGEN_PRIVATE_CONF=./private_config.yaml uv run train --base-config ./config/era5_local.yml \
  --options device=cpu with_mixed_precision=False attention_dtype=fp32
```

### Multiple processes on one machine

Several GPUs, or several CPU processes, run with `torchrun`; use FSDP (`with_fsdp=True`), since
plain DDP (`with_fsdp: False`, the default in `config/era5_local.yml`) currently fails in the
backward pass:

```bash
WEATHERGEN_PRIVATE_CONF=./private_config.yaml uv run torchrun --standalone --local-addr 127.0.0.1 \
  --nproc-per-node 2 --no-python train --base-config ./config/era5_local.yml --options with_fsdp=True
```

`--local-addr 127.0.0.1` is needed on macOS: without it, torchrun derives the address of the
processes from the host name, which can resolve to an unreachable address (e.g. on a VPN).
