// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open Dfx.Util

[<TestFixture>]
type TestRandom () =

    [<Test>]
    member _.TestRandomSeed () =
        Random.Seed(1)
        let a1 = Random.Uniform()
        Random.Seed(1)
        let a2 = Random.Uniform()
        let a3 = Random.Uniform()

        Assert.AreEqual(a1, a2)
        Assert.AreNotEqual(a2, a3)

    [<Test>]
    member _.TestRandomNormal () =
        // Normal samples are generated in pairs; the second of a pair is returned by the next call
        Random.Seed(1)
        let n = 200000
        let samples = Array.init n (fun _ -> Random.Normal())
        let mean = Array.average samples
        let var = samples |> Array.averageBy (fun v -> (v - mean) * (v - mean))
        // correlation between consecutive samples, which includes the two samples of each pair
        let lag1 = (Array.init (n - 1) (fun i -> (samples[i] - mean) * (samples[i + 1] - mean)) |> Array.average) / var
        Assert.True(abs mean < 0.01)
        Assert.True(abs (var - 1.0) < 0.02)
        Assert.True(abs lag1 < 0.01)

    [<Test>]
    member _.TestRandomNormalSeed () =
        // Setting the seed discards a pending second sample, so the sequence after a seed is always the same
        Random.Seed(1)
        let a1 = Random.Normal()
        Random.Seed(1)
        let a2 = Random.Normal()
        let a3 = Random.Normal()
        Random.Seed(1)
        let a4 = Random.Normal()
        Assert.AreEqual(a1, a2)
        Assert.AreEqual(a1, a4)
        Assert.AreNotEqual(a2, a3)

    [<Test>]
    member _.TestRandomUUID () =
        Random.Seed(1)
        let a1 = Random.UUID()
        Random.Seed(1)
        let a2 = Random.UUID()
        let a3 = Random.UUID()

        Assert.AreEqual(a1, a2)
        Assert.AreNotEqual(a2, a3)
