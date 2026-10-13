# dfx - Development Guide

You can clone this repository to your machine as follows:
```
git clone https://github.com/dfx-org/dfx.git
cd dfx
```

## Run tests

Required:
- Install the [.NET SDK](https://dotnet.microsoft.com/download) (version 10 or later) for your system

Use the following command in the root directory of this repository:
```
dotnet test --logger "console;verbosity=detailed"
```

## Build dfx in Docker

Required:
- Install [Docker](https://hub.docker.com/search/?type=edition&offering=community) for your system

Build a Docker image called `dfx`. This will work without any local .NET installation and build dfx inside the image.
```
docker build -t dfx .
```

Use the following to instantiate a Docker container from the `dfx` image and run the tests inside:
```
docker run --rm dfx dotnet test --logger "console;verbosity=detailed"
```

## Building against locally built TorchSharp packages

To add features you may have extend TorchSharp to make extra features of LibTorch available.

The build is set up to look for a parallel build of TorchSharp, e.g.

    C:\GitHub\dsyme\dfx
    C:\GitHub\dsyme\TorchSharp

To build, test and pack TorchSharp in that repo do this:

    .\build build
    .\build test
    .\build pack

You will see something like this

    Successfully created package 'C:\GitHub\dsyme\TorchSharp\bin/packages/Debug/TorchSharp.0.3.0-local-Debug-20200520.nupkg'.
    Successfully created package 'C:\GitHub\dsyme\TorchSharp\bin/packages/Debug/LibTorch.Redist.0.3.0-local-Debug-20200520.nupkg'.

with warning:

    warning : Packages will be incomplete and unusable on other platforms...

To consume the packages into dfx adjust TorchSharpVersion in Directory.Build.props.

When rebuilding the TorchSharp you will need to clear your package cache to pick up the new nuget package with the same version id, e.g.

    rmdir /q /s %USERPROFILE%\.nuget\packages\torchsharp
    rmdir /q /s %USERPROFILE%\.nuget\packages\LibTorch.Redist
    dotnet restore

The LibTorch packages are quite large and you may need to watch disk space.

## The Reference Backend

The "Reference" backend defines the semantics we expect of the Torch backend.

Sometimes configurations of Torch expose small differences in semantics (e.g. when using CUDA, or functionality not suppored for integer tensors).  We generally seek to paper
over those cracks by working around the problems in the Torch backend. 

## Developing and Testing on GPU

By default in-branch testing is only done on CPU.  To enable on GPU/CUDA you must:

1. Make sure you have a CUDA-capable NVIDIA GPU with a recent driver installed. The TorchSharp CUDA packages include the CUDA runtime (currently CUDA 12.8, Windows and Linux).

2. Use `dotnet test /p:DFX_TESTGPU=true --logger "console;verbosity=detailed"`, or set the environment variable `DFX_TESTGPU=true`

3. Verify that `dfx.isCudaAvailable()` returns true. The tests run on every available device.

## Micro Performance Benchmarking 

Python numbers must be collected in a separate run, they are currently injected back into source code (ugh)
to get figures in one report.  There are better ways to do this.

To update Python benchmarks on your machine (note, writes back results into source code)

    dotnet run --project tests\Dfx.Benchmarks.Python\Dfx.Benchmarks.Python.fsproj -c Release --filter "*"

This takes a while to run.

To run benchmarks:

    dotnet run --project tests\Dfx.Benchmarks\Dfx.Benchmarks.fsproj -c Release --filter "*"

To filter etc., see `--help`

## Testing on macOS (Apple Silicon)

The Apple Silicon build of libtorch loads OpenMP (`libomp`) from Homebrew's install location, so install it before running the tests:

    brew install libomp
