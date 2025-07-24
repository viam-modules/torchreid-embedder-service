# torchreid-embedder-service
An person embedder for features based person tracking


## Makefile targets for arm-jetson JP6 machines only

This project includes a `Makefile` script to automate the PyInstaller build process for Jetson machines. Building and deploying the module for other platforms should be done through CI.
PyInstaller is used to create standalone executables from the Python module scripts.

## Makefile targets for arm-jetson JP6 machines only

This project includes a `Makefile` script to automate the PyInstaller build process for Jetson machines. Building and deploying the module for other platforms should be done through CI.
PyInstaller is used to create standalone executables from the Python module scripts.

####  1. `make setup-jp6`

1. installs system dependencies (cuDNN and cuSPARSELt)
2. creates venv environment (under `./build/.venv`)
3. gets/builds python packages wheel files - Torch, Torchvision (built from source)

Cleaned with `make clean` (this also deletes pyinstaller build directory)

#### 2.  `make module.tar.gz`
This command builds the module executable using PyInstaller and creates a `.tar` file with the files needed to run the module.
The PyInstaller executable is created under `./build/pyinstaller_dist`.

#### 3. Upload to the registry with the right tag

```bash
viam login
viam module upload --version 0.0.0-rc0 --platform linux/arm64 --tags 'jetpack:6' archive.tar.gz
```

Cleaned with `make clean`
