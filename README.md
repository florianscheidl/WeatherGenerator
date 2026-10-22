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

This setup is WIP. This manual only works for machines compatible with torch 2.9.1 and cuda 12.9. Moreover, it only works for a single dataset configuration referenced below.

### Installation and setup

1. Install uv, see https://docs.astral.sh/uv/getting-started/installation/.
2. Clone the repo and cd to `WeatherGenerator`
3. Create output directories and run sync script:
    ```bash
    mkdir -p logs models output results
    ./scripts/actions.sh sync
    ```

###  Download data
- ERA5, 2020, 1-month.
    ```bash
    uv run --with "anemoi-datasets[remote]" anemoi-datasets create --overwrite datasets/download_configs/era5_o96_2020_1m.yaml datasets/era5-o96-2020-1pct-6h-v1.zarr
    ```


### Training

```bash
WEATHERGEN_PRIVATE_CONF=./local_config.yml uv run train --base-config ./config/era5_local.yml
```

### Inference on a small GPU

To run inference on a GPU with less memory than the model needs in full precision, add these
options to the inference command, e.g. for run `mz7oyrr5`:
```bash
WEATHERGEN_PRIVATE_CONF=$(pwd)/private_config.yaml uv run inference --from-run-id mz7oyrr5 \
  --options model_param_dtype=bf16 model_offload_blocks=true model_offload_inputs=true \
    streams.ERA5.max_num_targets=20000
```
- `model_param_dtype=bf16` loads the model on the CPU and casts its parameters to bf16 before
  moving it to the GPU.
- `model_offload_blocks=true` keeps the parameters of the large blocks (transformer blocks,
  per-stream embeddings) in CPU memory and copies each block to the GPU only for its forward
  pass. This is slower, since all parameters are copied to the GPU in each forward pass.
- `model_offload_inputs=true` moves the input data to the GPU stream by stream, when it is
  embedded, instead of all at once.
- `streams.ERA5.max_num_targets=20000` predicts the `ERA5` stream at a random subset of 20000
  grid points instead of all of them, since the decoder memory grows with the number of points.
  Use the name of a stream of your run with many target points.

With these options, inference of one sample with one forecast step of `mz7oyrr5` (about 1B
parameters) used 3.5 GB of GPU memory. The first three options are not supported with DDP,
i.e. on more than one GPU.
