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

### Inference with data from the JSC data catalog

The datasets listed in the [JSC data catalog](https://esm-data.fz-juelich.de/catalog/gui/v1/stac-browser/)
can be read directly over HTTPS, without downloading them first. Dataset filenames are
looked up in `data_paths`, which can contain URLs as well as local directories.

1. Put the checkpoint and the model config of the trained run in
   `<path_shared_working_dir>/models/<run_id>/`, where `path_shared_working_dir` is set in
   `private_config.yaml`. For run `mz7oyrr5` with the default `private_config.yaml`:
    ```
    shared_working_dir/models/mz7oyrr5/mz7oyrr5_latest.chkpt
    shared_working_dir/models/mz7oyrr5/model_mz7oyrr5_latest.json
    ```
2. Run inference from the root of the repository, with
   [config/jsc_catalog_data.yml](config/jsc_catalog_data.yml) to read the data from the
   catalog and [config/inference_one_sample.yml](config/inference_one_sample.yml) to forecast
   a single sample:
    ```bash
    WEATHERGEN_PRIVATE_CONF=$(pwd)/private_config.yaml uv run inference \
      --from-run-id mz7oyrr5 \
      --config config/jsc_catalog_data.yml config/inference_one_sample.yml
    ```
   The forecast is written to `shared_working_dir/results/<new run_id>/`.

`config/inference_one_sample.yml` runs one sample with a single forecast step (+6h) from
2023-03-01. Change `test_config` for other dates or more forecast steps. Note that:
- Downloading the data of one sample takes about 2 minutes.
- The Metop-B IASI dataset of run `mz7oyrr5` is not in the catalog, so it is left out.
  Choose dates covered by the other IASI datasets (Metop-A until 2021, Metop-C from
  2019-12 to 2023-05), else the IASI input is missing.
