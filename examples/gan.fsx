#!/usr/bin/env -S dotnet fsi

#I "../tests/Dfx.Tests/bin/Debug/net10.0"
#r "Dfx.Core.dll"
#r "Dfx.Data.dll"
#r "Dfx.Backends.Torch.dll"

// Libtorch binaries
// Option A: you can use a platform-specific nuget package
#r "nuget: TorchSharp-cpu, 0.107.0"
// #r "nuget: TorchSharp-cuda-linux, 0.107.0"
// #r "nuget: TorchSharp-cuda-windows, 0.107.0"
// Option B: you can use a local libtorch installation
// System.Runtime.InteropServices.NativeLibrary.Load("/home/gunes/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch.so")


open Dfx
open Dfx.Compose
open Dfx.Model
open Dfx.Data
open Dfx.Optim
open Dfx.Util

dfx.config(backend=Backend.Torch, device=Device.CPU)
dfx.seed(4)

let nz = 128

// PyTorch style
// type Generator(nz:int) =
//     inherit Model()
//     let fc1 = Linear(nz, 256)
//     let fc2 = Linear(256, 512)
//     let fc3 = Linear(512, 1024)
//     let fc4 = Linear(1024, 28*28)
//     do base.add([fc1; fc2; fc3; fc4])
//     override self.forward(x) =
//         x
//         |> dfx.view([-1;nz])
//         |> fc1.forward
//         |> dfx.leakyRelu(0.2)
//         |> fc2.forward
//         |> dfx.leakyRelu(0.2)
//         |> fc3.forward
//         |> dfx.leakyRelu(0.2)
//         |> fc4.forward
//         |> dfx.tanh
// type Discriminator(nz:int) =
//     inherit Model()
//     let fc1 = Linear(28*28, 1024)
//     let fc2 = Linear(1024, 512)
//     let fc3 = Linear(512, 256)
//     let fc4 = Linear(256, 1)
//     do base.add([fc1; fc2; fc3; fc4])
//     override self.forward(x) =
//         x
//         |> dfx.view([-1;28*28])
//         |> fc1.forward
//         |> dfx.leakyRelu(0.2)
//         |> dfx.dropout(0.3)
//         |> fc2.forward
//         |> dfx.leakyRelu(0.2)
//         |> dfx.dropout(0.3)
//         |> fc3.forward
//         |> dfx.leakyRelu(0.2)
//         |> dfx.dropout(0.3)
//         |> fc4.forward
//         |> dfx.sigmoid
// let generator = Generator(nz)
// let discriminator = Discriminator(nz)

// dfx compositional style
let generator =
    dfx.view([-1;nz])
    --> Linear(nz, 256)
    --> dfx.leakyRelu(0.2)
    --> Linear(256, 512)
    --> dfx.leakyRelu(0.2)
    --> Linear(512, 1024)
    --> dfx.leakyRelu(0.2)
    --> Linear(1024, 28*28)
    --> dfx.tanh

let discriminator =
    dfx.view([-1; 28*28])
    --> Linear(28*28, 1024)
    --> dfx.leakyRelu(0.2)
    --> dfx.dropout(0.3)
    --> Linear(1024, 512)
    --> dfx.leakyRelu(0.2)
    --> dfx.dropout(0.3)
    --> Linear(512, 256)
    --> dfx.leakyRelu(0.2)
    --> dfx.dropout(0.3)
    --> Linear(256, 1)
    --> dfx.sigmoid

printfn "Generator\n%s" (generator.summary())

printfn "Discriminator\n%s" (discriminator.summary())

let epochs = 10
let batchSize = 16
let validInterval = 100

let urls = ["https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"]

let mnist = MNIST("../data", urls=urls, train=true, transform=fun t -> (t - 0.5) / 0.5)
let loader = mnist.loader(batchSize=batchSize, shuffle=true)

let gopt = Adam(generator, lr=dfx.tensor(0.0002), beta1=dfx.tensor(0.5))
let dopt = Adam(discriminator, lr=dfx.tensor(0.0002), beta1=dfx.tensor(0.5))

let fixedNoise = dfx.randn([batchSize; nz])

let glosses = ResizeArray()
let dlosses = ResizeArray()
let dxs = ResizeArray()
let dgzs = ResizeArray()

let start = System.DateTime.Now
for epoch = 1 to epochs do
    for i, x, _ in loader.epoch() do
        let labelReal = dfx.ones([batchSize; 1])
        let labelFake = dfx.zeros([batchSize; 1])

        // update discriminator
        generator.noDiff()
        discriminator.reverseDiff()

        let doutput = x --> discriminator
        let dx = doutput.mean() |> float
        let dlossReal = dfx.bceLoss(doutput, labelReal)

        let z = dfx.randn([batchSize; nz])
        let goutput = z --> generator
        let doutput = goutput --> discriminator
        let dgz = doutput.mean() |> float
        let dlossFake = dfx.bceLoss(doutput, labelFake)

        let dloss = dlossReal + dlossFake
        dloss.reverse()
        dopt.step()
        dlosses.Add(float dloss)
        dxs.Add(float dx)
        dgzs.Add(float dgz)

        // update generator
        generator.reverseDiff()
        discriminator.noDiff()

        let goutput = z --> generator
        let doutput = goutput --> discriminator
        let gloss = dfx.bceLoss(doutput, labelReal)
        gloss.reverse()
        gopt.step()
        glosses.Add(float gloss)

        printfn "%A Epoch: %A/%A minibatch: %A/%A gloss: %A dloss: %A d(x): %A d(g(z)): %A" (System.DateTime.Now - start) epoch epochs (i+1) loader.length (float gloss) (float dloss) dx dgz

        if i % validInterval = 0 then
            let realFileName = sprintf "gan_real_samples_epoch_%A_minibatch_%A.png" epoch (i+1)
            printfn "Saving real samples to %A" realFileName
            ((x+1)/2).saveImage(realFileName, normalize=false)
            let fakeFileName = sprintf "gan_fake_samples_epoch_%A_minibatch_%A.png" epoch (i+1)
            printfn "Saving fake samples to %A" fakeFileName
            let goutput = fixedNoise --> generator
            ((goutput.view([-1;1;28;28])+1)/2).saveImage(fakeFileName, normalize=false)

            let plt = Pyplot()
            plt.plot(glosses |> dfx.tensor, label="Generator")
            plt.plot(dlosses |> dfx.tensor, label="Discriminator")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.tightLayout()
            plt.savefig (sprintf "gan_loss_epoch_%A_minibatch_%A.pdf" epoch (i+1))

            let plt = Pyplot()
            plt.plot(dxs |> dfx.tensor, label="d(x)")
            plt.plot(dgzs |> dfx.tensor, label="d(g(z))")
            plt.xlabel("Iterations")
            plt.ylabel("Score")
            plt.legend()
            plt.tightLayout()
            plt.savefig (sprintf "gan_score_epoch_%A_minibatch_%A.pdf" epoch (i+1))            