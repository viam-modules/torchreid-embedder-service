# torchreid-embedder-service
An person embedder for features based person tracking


## Configure your `torchreid-embedder-service` vision service

```json
{
  "model_path": "/path/to/your/model"
}
```


## Run test

```bash
make setup #or make setup-jp6 on jetson machines
source build/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)\
python src/test_integration.py
```





## Makefile targets for arm-jetson JP6 machines only

This project includes a `Makefile` script to automate the PyInstaller build process for Jetson machines. Building and deploying the module for other platforms should be done through CI.
PyInstaller is used to create standalone executables from the Python module scripts.

####  1. Build dev environment
Run :
```bash
make setup-jp6
```

1. installs system dependencies (cuDNN and cuSPARSELt)
2. creates venv environment (under `./build/.venv`)
3. gets/builds python packages wheel files - Torch, Torchvision (built from source)

Cleaned with `make clean` (this also deletes pyinstaller build directory)

#### 2. Build module executable

```bash
make pyinstaller
```
This command builds the module executable using PyInstaller.

This creates the PyInstaller executable under `./build/pyinstaller_dist`.

#### 3. Upload to viam registry:

Edit  `first_run` field in `meta.json` file
```
"first_run": "first_run_jp6.sh"
```


Compress the module with the updated `meta.json` and `first_run_jp6.sh` using:
```bash
make module.tar.gz
```

and upload to the registry:

```bash
viam login
viam module upload --version 0.0.0-rc0 --platform linux/arm64 --tags 'jetpack:6' module.tar.gz
```

Cleaned with `make clean`
