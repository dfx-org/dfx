// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open Dfx
open Dfx.Compose
open Dfx.Model
open Dfx.Data
open Dfx.Optim


[<TestFixture>]
type TestModelVAE() =
    
    [<Test>]
    member _.TestModelVAEMLP () =
        // Fits a little VAEMLP to structured noise
        let xdim, zdim, n = 8, 4, 16
        let m = VAEMLP(xdim*xdim, zdim)
        let x = dfx.stack(Array.init n (fun _ -> dfx.eye(xdim)*dfx.rand([xdim;xdim])))

        let lr, steps = 1e-2, 50
        let optimizer = Adam(m, lr=dfx.tensor(lr))
        let loss0 = float <| m.loss(x)
        let mutable loss = loss0
        for _ in 0..steps do
            m.reverseDiff()
            let l = m.loss(x)
            l.reverse()
            optimizer.step()
            loss <- float l

        Assert.Less(loss, loss0/2.)
    
    [<Test>]
    member _.TestModelVAE () =
        // Fits a little VAE to structured noise
        let xdim, zdim, n = 28, 4, 16
        let encoder = dfx.flatten(1) --> Linear(xdim*xdim, 8) --> dfx.relu
        let decoder = Linear(8, xdim*xdim) --> dfx.sigmoid

        let m = VAE([xdim;xdim], zdim, encoder, decoder)
        let x = dfx.stack(Array.init n (fun _ -> dfx.eye(xdim)*dfx.rand([xdim;xdim])))

        let lr, steps = 1e-2, 25
        let optimizer = Adam(m, lr=dfx.tensor(lr))
        let loss0 = float <| m.loss(x)
        let mutable loss = loss0
        for _ in 0..steps do
            m.reverseDiff()
            let l = m.loss(x)
            l.reverse()
            optimizer.step()
            loss <- float l

        Assert.Less(loss, loss0/2.)

    [<Test>]
    member _.TestModelVAEMLPSaveLoadState () =
        let xdim, zdim, n = 8, 4, 16
        let net = VAEMLP(xdim*xdim, zdim)

        let fileName = System.IO.Path.GetTempFileName()
        dfx.save(net.state, fileName) // Save pre-use
        let _ = dfx.randn([n; xdim*xdim]) --> net // Use
        net.state <- dfx.load(fileName) // Load after-use

        Assert.True(true)

    [<Test>]
    member _.TestModelVAESaveLoadState () =
        let xdim, zdim, n = 28, 4, 16
        let encoder = dfx.flatten(1) --> Linear(xdim*xdim, 8) --> dfx.relu
        let decoder = Linear(8, xdim*xdim) --> dfx.sigmoid
        let net = VAE([xdim;xdim], zdim, encoder, decoder)

        let fileName = System.IO.Path.GetTempFileName()
        dfx.save(net.state, fileName) // Save pre-use
        let _ = dfx.randn([n; xdim; xdim]) --> net // Use
        net.state <- dfx.load(fileName) // Load after-use

        Assert.True(true)