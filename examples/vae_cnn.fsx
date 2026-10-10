#!/usr/bin/env -S dotnet fsi

#I "../tests/Dfx.Tests/bin/Debug/net6.0"
#r "Dfx.Core.dll"
#r "Dfx.Data.dll"
#r "Dfx.Backends.Torch.dll"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
#r "nuget: TorchSharp-cpu, 0.96.5"
// #r "nuget: TorchSharp-cuda-linux, 0.96.5"
// #r "nuget: TorchSharp-cuda-windows, 0.96.5"
// Option B: you can use a local libtorch installation
// System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open Dfx
open Dfx.Compose
open Dfx.Model
open Dfx.Optim
open Dfx.Data


dfx.config(backend=Backend.Torch, device=Device.CPU)
dfx.seed(0)

let epochs = 2
let batchSize = 32
let validInterval = 250
let numSamples = 32

let urls = ["https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"]

let trainSet = MNIST("../data", urls=urls, train=true, transform=id)
let trainLoader = trainSet.loader(batchSize=batchSize, shuffle=true)
let validSet = MNIST("../data", urls=urls, train=false, transform=id)
let validLoader = validSet.loader(batchSize=batchSize, shuffle=false)

let encoder =
    Conv2d(1, 32, 4, 2)
    --> dfx.relu
    --> Conv2d(32, 64, 4, 2)
    --> dfx.relu
    --> Conv2d(64, 128, 4, 2)
    --> dfx.flatten(1)

let decoder =
    dfx.unflatten(1, [128;1;1])
    --> ConvTranspose2d(128, 64, 4, 2)
    --> dfx.relu
    --> ConvTranspose2d(64, 32, 4, 3)
    --> dfx.relu
    --> ConvTranspose2d(32, 1, 4, 2)
    --> dfx.sigmoid

let model = VAE([1;28;28], 64, encoder, decoder)

printfn "Model\n%s" (model.summary())

let optimizer = Adam(model, lr=dfx.tensor(0.001))

for epoch = 1 to epochs do
    for i, x, _ in trainLoader.epoch() do
        model.reverseDiff()
        let l = model.loss(x)
        l.reverse()
        optimizer.step()
        printfn "Epoch: %A/%A minibatch: %A/%A loss: %A" epoch epochs i trainLoader.length (float(l))

        if i % validInterval = 0 then
            let mutable validLoss = dfx.zero()
            for _, x, _ in validLoader.epoch() do
                validLoss <- validLoss + model.loss(x, normalize=false)
            validLoss <- validLoss / validSet.length
            printfn "Validation loss: %A" (float validLoss)
            let fileName = sprintf "vae_cnn_samples_epoch_%A_minibatch_%A.png" epoch i
            printfn "Saving %A samples to %A" numSamples fileName
            let samples = model.sample(numSamples).view([-1; 1; 28; 28])
            samples.saveImage(fileName)

