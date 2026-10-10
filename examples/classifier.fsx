#!/usr/bin/env -S dotnet fsi

#I "../tests/Dfx.Tests/bin/Debug/net10.0"
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
open Dfx.Model
open Dfx.Compose
open Dfx.Optim
open Dfx.Data
open Dfx.Util

dfx.config(backend=Backend.Torch, device=Device.CPU)
dfx.seed(0)

// PyTorch style
// type Classifier() =
//     inherit Model()
//     let conv1 = Conv2d(1, 32, 3, 2)
//     let conv2 = Conv2d(32, 64, 3, 2)
//     let fc1 = Linear(576, 128)
//     let fc2 = Linear(128, 10)
//     do base.add([conv1; conv2; fc1; fc2])
//     override self.forward(x) =
//         x
//         |> conv1.forward
//         |> dfx.relu
//         |> conv2.forward
//         |> dfx.relu
//         |> dfx.maxpool2d(2)
//         |> dfx.dropout(0.25)
//         |> dfx.flatten(1)
//         |> fc1.forward
//         |> dfx.relu
//         |> dfx.dropout(0.5)
//         |> fc2.forward
//         |> dfx.logsoftmax(dim=1)
// let classifier = Classifier()

// Dfx compositional style
let classifier =
    Conv2d(1, 32, 3, 2)
    --> dfx.relu
    --> Conv2d(32, 64, 3, 2)
    --> dfx.relu
    --> dfx.maxpool2d(2)
    --> dfx.dropout(0.25)
    --> dfx.flatten(1)
    --> Linear(576, 128)
    --> dfx.relu
    --> dfx.dropout(0.5)
    --> Linear(128, 10)
    --> dfx.logsoftmax(dim=1)

let epochs = 20
let batchSize = 64
let numSamples = 4

let urls = ["https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"]

let trainSet = MNIST("../data", urls=urls, train=true)
let trainLoader = trainSet.loader(batchSize=batchSize, shuffle=true)
let validSet = MNIST("../data", urls=urls, train=false)
let validLoader = validSet.loader(batchSize=batchSize, shuffle=false)


printfn "Model:\n%s" (classifier.summary())

let optimizer = Adam(classifier, lr=dfx.tensor(0.001))

for epoch = 1 to epochs do
    for i, data, target in trainLoader.epoch() do
        classifier.reverseDiff()
        let output = data --> classifier
        let l = dfx.nllLoss(output, target)
        l.reverse()
        optimizer.step()
        if i % 10 = 0 then
            printfn "Epoch: %A/%A, minibatch: %A/%A, loss: %A" epoch epochs i trainLoader.length (float(l))


    printfn "Computing validation loss"
    classifier.noDiff()
    let mutable validLoss = dfx.zero()
    let mutable correct = 0
    for j, data, target in validLoader.epoch() do
        let output = data --> classifier
        validLoss <- validLoss + dfx.nllLoss(output, target, reduction="sum")
        let pred = output.argmax(1)
        correct <- correct + int (pred.eq(target).sum())
    validLoss <- validLoss / validSet.length
    let accuracy = 100.*(float correct) / (float validSet.length)
    printfn "\nValidation loss: %A, accuracy: %.2f%%" (float validLoss) accuracy

    let samples, sampleLabels = validLoader.batch(numSamples)
    printfn "Sample predictions:\n%s" (samples.toImageString(gridCols=4))
    printfn "True labels     : %A " (sampleLabels.int())
    let predictedLabels = (samples --> classifier).argmax(dim=1)
    printfn "Predicted labels: %A\n" predictedLabels

