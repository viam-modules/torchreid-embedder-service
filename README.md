# torchreid-embedder-service
An person embedder for features based person tracking


## Makefile targets for arm-jetson JP6 machines only

This project includes a `Makefile` script to automate the PyInstaller build process for Jetson machines. Building and deploying the module for other platforms should be done through CI.
PyInstaller is used to create standalone executables from the Python module scripts.

####  `make setup-jp6`

1. installs system dependencies (cuDNN and cuSPARSELt)
2. creates venv environment (under `./build/.venv`)
3. gets/builds python packages wheel files - Torch, Torchvision (built from source)

Cleaned with `make clean` (this also deletes pyinstaller build directory)

#### `make pyinstaller`
This command builds the module executable using PyInstaller.

This creates the PyInstaller executable under `./build/pyinstaller_dist`.
To upload to viam registry:

First copy `./build/pyinstaller_dist/main` in the `camera-object-tracking-service` repository.

```bash
cd camera-object-tracking-service
cp ./build/pyinstaller_dist/main ./
```

Compress and upload to the registry:

```bash
viam login
tar -czvf archive.tar.gz meta.json main first_run.sh  #needs to be on the same level
viam module upload --version 0.0.0-rc0 --platform linux/arm64 --tags 'jetpack:6' archive.tar.gz
```

Cleaned with `make clean-pyinstaller`
