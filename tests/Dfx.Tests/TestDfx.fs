// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open Dfx
open Dfx.Compose
open Dfx.Shorten
open Dfx.Numerical
open Dfx.Numerical.Shorten
open Dfx.Model


[<TestFixture>]
type TestDfx () =

    let rosenbrock (x:Tensor) =
        let x, y = x[0], x[1]
        (1. - x)**2 + 100. * (y - x**2)**2
    let rosenbrockGrad (x:Tensor) =
        let x, y = x[0], x[1]
        dfx.tensor([-2*(1-x)-400*x*(-(x**2) + y); 200*(-(x**2) + y)])
    let rosenbrockHessian (x:Tensor) =
        let x, y = x[0], x[1]
        dfx.tensor([[2.+1200.*x*x-400.*y, -400.*x],[-400.*x, 200.*dfx.one()]])

    let fscalarscalar (x:Tensor) = dfx.sin x
    let fscalarscalarDiff (x:Tensor) = dfx.cos x

    let fscalarvect3 (x:Tensor) = dfx.stack([sin x; exp x; cos x])
    let fscalarvect3Diff (x:Tensor) = dfx.stack([cos x; exp x; -sin x])
    let fscalarvect3Diff2 (x:Tensor) = dfx.stack([-sin x; exp x; -cos x])
    let fscalarvect3Diff3 (x:Tensor) = dfx.stack([-cos x; exp x; sin x])

    let fvect2vect2 (x:Tensor) =
        let x, y = x[0], x[1]
        dfx.stack([x*x*y; 5*x+sin y])
    let fvect2vect2Jacobian (x:Tensor) =
        let x, y = x[0], x[1]
        dfx.tensor([[2*x*y; x*x];[dfx.tensor(5.); cos y]])

    let fvect3vect2 (x:Tensor) =
        let x, y, z = x[0], x[1], x[2]
        dfx.stack([x*y+2*y*z;2*x*y*y*z])
    let fvect3vect2Jacobian (x:Tensor) =
        let x, y, z = x[0], x[1], x[2]
        dfx.tensor([[y;x+2*z;2*y];[2*y*y*z;4*x*y*z;2*x*y*y]])

    let fvect3vect3 (x:Tensor) =
        let r, theta, phi = x[0], x[1], x[2]
        dfx.stack([r*(sin phi)*(cos theta); r*(sin phi)*(sin theta); r*cos phi])
    let fvect3vect3Jacobian (x:Tensor) =
        let r, theta, phi = x[0], x[1], x[2]
        dfx.tensor([[(sin phi)*(cos theta); -r*(sin phi)*(sin theta); r*(cos phi)*(cos theta)];[(sin phi)*(sin theta); r*(sin phi)*(cos theta); r*(cos phi)*(sin theta)];[cos phi; dfx.zero(); -r*sin phi]])

    let fvect3vect4 (x:Tensor) =
        let y1, y2, y3, y4 = x[0], 5*x[2], 4*x[1]*x[1]-2*x[2],x[2]*sin x[0]
        dfx.stack([y1;y2;y3;y4])
    let fvect3vect4Jacobian (x:Tensor) =
        let z, o = dfx.zero(), dfx.one()
        dfx.tensor([[o,z,z],[z,z,5*o],[z,8*x[1],-2*o],[x[2]*cos x[0],z,sin x[0]]])

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestZero () =
        let t = dfx.zero(dtype=Int32)
        let tCorrect = dfx.tensor(0)
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestZeros () =
        let t = dfx.zeros([2;3], dtype=Int32)
        let tCorrect = dfx.tensor([[0,0,0],[0,0,0]])
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestOne () =
        let t = dfx.one(dtype=Int32)
        let tCorrect = dfx.tensor(1)
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestOnes () =
        let t = dfx.ones([2;3], dtype=Int32)
        let tCorrect = dfx.tensor([[1,1,1],[1,1,1]])
        Assert.CheckEqual(tCorrect, t)

    [<Test>]
    member this.TestRand () =
        let t = dfx.rand([1000])
        let tMean = t.mean()
        let tMeanCorrect = dfx.tensor(0.5)
        let tStddev = t.std()
        let tStddevCorrect = dfx.tensor(1./12.) |> dfx.sqrt
        Assert.True(tMeanCorrect.allclose(tMean, 0.1))
        Assert.True(tStddevCorrect.allclose(tStddev, 0.1))

    [<Test>]
    member this.TestRandn () =
        let t = dfx.randn([1000])
        let tMean = t.mean()
        let tMeanCorrect = dfx.tensor(0.)
        let tStddev = t.std()
        let tStddevCorrect = dfx.tensor(1.)
        Assert.True(tMeanCorrect.allclose(tMean, 0.1, 0.1))
        Assert.True(tStddevCorrect.allclose(tStddev, 0.1, 0.1))

    [<Test>]
    member this.TestArange () =
        let t1 = dfx.arange(5.)
        let t1Correct = dfx.tensor([0.,1.,2.,3.,4.])
        let t2 = dfx.arange(startVal=1., endVal=4.)
        let t2Correct = dfx.tensor([1.,2.,3.])
        let t3 = dfx.arange(startVal=1., endVal=2.5, step=0.5)
        let t3Correct = dfx.tensor([1.,1.5,2.])
        Assert.CheckEqual(t1Correct, t1)
        Assert.CheckEqual(t2Correct, t2)
        Assert.CheckEqual(t3Correct, t3)


    [<Test>]
    member this.TestSeed () =
        for combo in Combos.FloatingPointExcept16s do
            printfn "%A" (combo.device, combo.backend, combo.dtype)
            use _holder = dfx.useConfig(combo.dtype, combo.device, combo.backend)
            dfx.seed(123)
            let t = combo.randint(0,10,[25])
            dfx.seed(123)
            let t2 = combo.randint(0,10,[25])
            Assert.CheckEqual(t, t2)

    [<Test>]
    member this.TestSlice () =
        let t = dfx.tensor([1, 2, 3])
        let tSlice = t |> dfx.slice([0])
        let tSliceCorrect = t[0]
        Assert.CheckEqual(tSliceCorrect, tSlice)

    [<Test>]
    member this.TestDiff () =
        let x = dfx.tensor(1.5)
        let fx, d = dfx.fdiff fscalarvect3 x
        let d2 = dfx.diff fscalarvect3 x
        let nfx, nd = dfx.numfdiff 1e-5 fscalarvect3 x
        let nd2 = dfx.numdiff 1e-5 fscalarvect3 x
        let fxCorrect = fscalarvect3 x
        let dCorrect = fscalarvect3Diff x
        Assert.CheckEqual(fxCorrect, fx)
        Assert.CheckEqual(fxCorrect, nfx)
        Assert.CheckEqual(dCorrect, d)
        Assert.CheckEqual(dCorrect, d2)
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))

    [<Test>]
    member this.TestDiff2 () =
        let x = dfx.tensor(1.5)
        let fx, d = dfx.fdiff2 fscalarvect3 x
        let d2 = dfx.diff2 fscalarvect3 x
        let nfx, nd = dfx.numfdiff2 1e-2 fscalarvect3 x
        let nd2 = dfx.numdiff2 1e-2 fscalarvect3 x
        let fxCorrect = fscalarvect3 x
        let dCorrect = fscalarvect3Diff2 x
        Assert.CheckEqual(fxCorrect, fx)
        Assert.CheckEqual(fxCorrect, nfx)
        Assert.CheckEqual(dCorrect, d)
        Assert.CheckEqual(dCorrect, d2)
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))

    [<Test>]
    member this.TestDiffn () =
        let x = dfx.tensor(1.5)
        let fx, d = dfx.fdiffn 3 fscalarvect3 x
        let d2 = dfx.diffn 3 fscalarvect3 x
        let fxCorrect = fscalarvect3 x
        let dCorrect = fscalarvect3Diff3 x
        Assert.CheckEqual(fxCorrect, fx)
        Assert.CheckEqual(dCorrect, d)
        Assert.CheckEqual(dCorrect, d2)

    [<Test>]
    member this.TestGrad () =
        let x = dfx.tensor([1.5;2.5])
        let fx1, g1 = dfx.fgrad rosenbrock x
        let fx2, g2 = dfx.fg rosenbrock x
        let g3 = dfx.grad rosenbrock x
        let g4 = dfx.g rosenbrock x
        let nfx1, ng1 = dfx.numfgrad 1e-6 rosenbrock x
        let nfx2, ng2 = dfx.numfg 1e-6 rosenbrock x
        let ng3 = dfx.numgrad 1e-6 rosenbrock x
        let ng4 = dfx.numg 1e-6 rosenbrock x
        let fxCorrect = rosenbrock x
        let gCorrect = rosenbrockGrad x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(gCorrect, g1)
        Assert.CheckEqual(gCorrect, g2)
        Assert.CheckEqual(gCorrect, g3)
        Assert.CheckEqual(gCorrect, g4)
        Assert.True(gCorrect.allclose(ng1, 0.1))
        Assert.True(gCorrect.allclose(ng2, 0.1))
        Assert.True(gCorrect.allclose(ng3, 0.1))
        Assert.True(gCorrect.allclose(ng4, 0.1))

    [<Test>]
    member this.TestGradScalarToScalar () =
        let x = dfx.tensor(1.5)
        let fx1, g1 = dfx.fgrad fscalarscalar x
        let fx2, g2 = dfx.fg fscalarscalar x
        let g3 = dfx.grad fscalarscalar x
        let g4 = dfx.g fscalarscalar x
        let nfx1, ng1 = dfx.numfgrad 1e-3 fscalarscalar x
        let nfx2, ng2 = dfx.numfg 1e-3 fscalarscalar x
        let ng3 = dfx.numgrad 1e-3 fscalarscalar x
        let ng4 = dfx.numg 1e-3 fscalarscalar x
        let fxCorrect = fscalarscalar x
        let gCorrect = fscalarscalarDiff x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(gCorrect, g1)
        Assert.CheckEqual(gCorrect, g2)
        Assert.CheckEqual(gCorrect, g3)
        Assert.CheckEqual(gCorrect, g4)
        Assert.True(gCorrect.allclose(ng1, 0.1))
        Assert.True(gCorrect.allclose(ng2, 0.1))
        Assert.True(gCorrect.allclose(ng3, 0.1))
        Assert.True(gCorrect.allclose(ng4, 0.1))

    [<Test>]
    member this.TestGradv () =
        let x = dfx.tensor([1.5;2.5])
        let v = dfx.tensor([2.75;-3.5])
        let fx1, gv1 = dfx.fgradv rosenbrock x v
        let fx2, gv2 = dfx.fgvp rosenbrock x v
        let gv3 = dfx.gradv rosenbrock x v
        let gv4 = dfx.gvp rosenbrock x v
        let nfx1, ngv1 = dfx.numfgradv 1e-5 rosenbrock x v
        let nfx2, ngv2 = dfx.numfgvp 1e-5 rosenbrock x v
        let ngv3 = dfx.numgradv 1e-5 rosenbrock x v
        let ngv4 = dfx.numgvp 1e-5 rosenbrock x v
        let fxCorrect = rosenbrock x
        let gvCorrect = dfx.dot(rosenbrockGrad x,  v)
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(gvCorrect, gv1)
        Assert.CheckEqual(gvCorrect, gv2)
        Assert.CheckEqual(gvCorrect, gv3)
        Assert.CheckEqual(gvCorrect, gv4)
        Assert.True(gvCorrect.allclose(ngv1, 0.1))
        Assert.True(gvCorrect.allclose(ngv2, 0.1))
        Assert.True(gvCorrect.allclose(ngv3, 0.1))
        Assert.True(gvCorrect.allclose(ngv4, 0.1))

    [<Test>]
    member this.TestJacobianv () =
        let x = dfx.tensor([1.5, 2.5, 3.])
        let v = dfx.tensor([2.75, -3.5, 4.])
        let fx1, jv1 = dfx.fjacobianv fvect3vect2 x v
        let fx2, jv2 = dfx.fjvp fvect3vect2 x v
        let jv3 = dfx.jacobianv fvect3vect2 x v
        let jv4 = dfx.jvp fvect3vect2 x v
        let nfx1, njv1 = dfx.numfjacobianv 1e-3 fvect3vect2 x v
        let nfx2, njv2 = dfx.numfjvp 1e-3 fvect3vect2 x v
        let njv3 = dfx.numjacobianv 1e-3 fvect3vect2 x v
        let njv4 = dfx.numjvp 1e-3 fvect3vect2 x v
        let fxCorrect = fvect3vect2 x
        let jvCorrect = dfx.matmul(fvect3vect2Jacobian x,  v.view([-1;1])).view(-1)
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(jvCorrect, jv1)
        Assert.CheckEqual(jvCorrect, jv2)
        Assert.CheckEqual(jvCorrect, jv3)
        Assert.CheckEqual(jvCorrect, jv4)
        Assert.True(jvCorrect.allclose(njv1, 0.1))
        Assert.True(jvCorrect.allclose(njv2, 0.1))
        Assert.True(jvCorrect.allclose(njv3, 0.1))
        Assert.True(jvCorrect.allclose(njv4, 0.1))

    [<Test>]
    member this.TestJacobianTv () =
        let x = dfx.tensor([1.5, 2.5, 3.])
        let v = dfx.tensor([2.75, -3.5])
        let fx, jTv = dfx.fjacobianTv fvect3vect2 x v
        let jTv2 = dfx.jacobianTv fvect3vect2 x v
        let fxCorrect = fvect3vect2 x
        let jTvCorrect = dfx.matmul(v.view([1;-1]), fvect3vect2Jacobian x).view(-1)
        Assert.CheckEqual(fxCorrect, fx)
        Assert.CheckEqual(jTvCorrect, jTv)
        Assert.CheckEqual(jTvCorrect, jTv2)

    [<Test>]
    member this.TestJacobian () =
        let x = dfx.arange(2.)
        let fx1, j1 = dfx.fjacobian fvect2vect2 x
        let fx2, j2 = dfx.fj fvect2vect2 x
        let j3 = dfx.jacobian fvect2vect2 x
        let j4 = dfx.j fvect2vect2 x
        let nfx1, nj1 = dfx.numfjacobian 1e-4 fvect2vect2 x
        let nfx2, nj2 = dfx.numfj 1e-4 fvect2vect2 x
        let nj3 = dfx.numjacobian 1e-4 fvect2vect2 x
        let nj4 = dfx.numj 1e-4 fvect2vect2 x
        let fxCorrect = fvect2vect2 x
        let jCorrect = fvect2vect2Jacobian x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(jCorrect, j1)
        Assert.CheckEqual(jCorrect, j2)
        Assert.CheckEqual(jCorrect, j3)
        Assert.CheckEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

        let x = dfx.arange(3.)
        let fx1, j1 = dfx.fjacobian fvect3vect2 x
        let fx2, j2 = dfx.fj fvect3vect2 x
        let j3 = dfx.jacobian fvect3vect2 x
        let j4 = dfx.j fvect3vect2 x
        let nfx1, nj1 = dfx.numfjacobian 1e-4 fvect3vect2 x
        let nfx2, nj2 = dfx.numfj 1e-4 fvect3vect2 x
        let nj3 = dfx.numjacobian 1e-4 fvect3vect2 x
        let nj4 = dfx.numj 1e-4 fvect3vect2 x
        let fxCorrect = fvect3vect2 x
        let jCorrect = fvect3vect2Jacobian x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(jCorrect, j1)
        Assert.CheckEqual(jCorrect, j2)
        Assert.CheckEqual(jCorrect, j3)
        Assert.CheckEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

        let x = dfx.arange(3.)
        let fx1, j1 = dfx.fjacobian fvect3vect3 x
        let fx2, j2 = dfx.fj fvect3vect3 x
        let j3 = dfx.jacobian fvect3vect3 x
        let j4 = dfx.j fvect3vect3 x
        let nfx1, nj1 = dfx.numfjacobian 1e-4 fvect3vect3 x
        let nfx2, nj2 = dfx.numfj 1e-4 fvect3vect3 x
        let nj3 = dfx.numjacobian 1e-4 fvect3vect3 x
        let nj4 = dfx.numj 1e-4 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let jCorrect = fvect3vect3Jacobian x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(jCorrect, j1)
        Assert.CheckEqual(jCorrect, j2)
        Assert.CheckEqual(jCorrect, j3)
        Assert.CheckEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

        let x = dfx.arange(3.)
        let fx1, j1 = dfx.fjacobian fvect3vect4 x
        let fx2, j2 = dfx.fj fvect3vect4 x
        let j3 = dfx.jacobian fvect3vect4 x
        let j4 = dfx.j fvect3vect4 x
        let nfx1, nj1 = dfx.numfjacobian 1e-4 fvect3vect4 x
        let nfx2, nj2 = dfx.numfj 1e-4 fvect3vect4 x
        let nj3 = dfx.numjacobian 1e-4 fvect3vect4 x
        let nj4 = dfx.numj 1e-4 fvect3vect4 x
        let fxCorrect = fvect3vect4 x
        let jCorrect = fvect3vect4Jacobian x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(jCorrect, j1)
        Assert.CheckEqual(jCorrect, j2)
        Assert.CheckEqual(jCorrect, j3)
        Assert.CheckEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

    [<Test>]
    member this.TestGradhessianv () =
        let x = dfx.tensor([1.5, 2.5])
        let v = dfx.tensor([0.5, -2.])
        let fx1, gv1, hv1 = dfx.fgradhessianv rosenbrock x v
        let fx2, gv2, hv2 = dfx.fghvp rosenbrock x v
        let gv3, hv3 = dfx.gradhessianv rosenbrock x v
        let gv4, hv4 = dfx.ghvp rosenbrock x v
        let fxCorrect = rosenbrock x
        let gvCorrect = dfx.dot(rosenbrockGrad x,  v)
        let hvCorrect = dfx.matmul(rosenbrockHessian x,  v.view([-1;1])).view(-1)
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(gvCorrect, gv1)
        Assert.CheckEqual(gvCorrect, gv2)
        Assert.CheckEqual(gvCorrect, gv3)
        Assert.CheckEqual(gvCorrect, gv4)
        Assert.CheckEqual(hvCorrect, hv1)
        Assert.CheckEqual(hvCorrect, hv2)
        Assert.CheckEqual(hvCorrect, hv3)
        Assert.CheckEqual(hvCorrect, hv4)

    [<Test>]
    member this.TestGradhessian () =
        let x = dfx.tensor([1.5, 2.5])
        let fx1, g1, h1 = dfx.fgradhessian rosenbrock x
        let fx2, g2, h2 = dfx.fgh rosenbrock x
        let g3, h3 = dfx.gradhessian rosenbrock x
        let g4, h4 = dfx.gh rosenbrock x
        let nfx1, ng1, nh1 = dfx.numfgradhessian 1e-3 rosenbrock x
        let nfx2, ng2, nh2 = dfx.numfgh 1e-3 rosenbrock x
        let ng3, nh3 = dfx.numgradhessian 1e-3 rosenbrock x
        let ng4, nh4 = dfx.numgh 1e-3 rosenbrock x
        let fxCorrect = rosenbrock x
        let gCorrect = rosenbrockGrad x
        let hCorrect = rosenbrockHessian x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(gCorrect, g1)
        Assert.CheckEqual(gCorrect, g2)
        Assert.CheckEqual(gCorrect, g3)
        Assert.CheckEqual(gCorrect, g4)
        Assert.CheckEqual(hCorrect, h1)
        Assert.CheckEqual(hCorrect, h2)
        Assert.CheckEqual(hCorrect, h3)
        Assert.CheckEqual(hCorrect, h4)
        Assert.True(gCorrect.allclose(ng1, 0.1))
        Assert.True(gCorrect.allclose(ng2, 0.1))
        Assert.True(gCorrect.allclose(ng3, 0.1))
        Assert.True(gCorrect.allclose(ng4, 0.1))
        Assert.True(hCorrect.allclose(nh1, 0.1))
        Assert.True(hCorrect.allclose(nh2, 0.1))
        Assert.True(hCorrect.allclose(nh3, 0.1))
        Assert.True(hCorrect.allclose(nh4, 0.1))

    [<Test>]
    member this.TestHessianv () =
        let x = dfx.tensor([1.5, 2.5])
        let v = dfx.tensor([0.5, -2.])
        let fx1, hv1 = dfx.fhessianv rosenbrock x v
        let fx2, hv2 = dfx.fhvp rosenbrock x v
        let hv3 = dfx.hessianv rosenbrock x v
        let hv4 = dfx.hvp rosenbrock x v
        let nfx1, nhv1 = dfx.numfhessianv 1e-3 rosenbrock x v
        let nfx2, nhv2 = dfx.numfhvp 1e-3 rosenbrock x v
        let nhv3 = dfx.numhessianv 1e-3 rosenbrock x v
        let nhv4 = dfx.numhvp 1e-3 rosenbrock x v
        let fxCorrect = rosenbrock x
        let hvCorrect = dfx.matmul(rosenbrockHessian x,  v.view([-1;1])).view(-1)
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(hvCorrect, hv1)
        Assert.CheckEqual(hvCorrect, hv2)
        Assert.CheckEqual(hvCorrect, hv3)
        Assert.CheckEqual(hvCorrect, hv4)
        Assert.True(hvCorrect.allclose(nhv1, 0.1))
        Assert.True(hvCorrect.allclose(nhv2, 0.1))
        Assert.True(hvCorrect.allclose(nhv3, 0.1))
        Assert.True(hvCorrect.allclose(nhv4, 0.1))

    [<Test>]
    member this.TestHessian () =
        let x = dfx.tensor([1.5, 2.5])
        let fx1, h1 = dfx.fhessian rosenbrock x
        let fx2, h2 = dfx.fh rosenbrock x
        let h3 = dfx.hessian rosenbrock x
        let h4 = dfx.h rosenbrock x
        let nfx1, nh1 = dfx.numfhessian 1e-3 rosenbrock x
        let nfx2, nh2 = dfx.numfh 1e-3 rosenbrock x
        let nh3 = dfx.numhessian 1e-3 rosenbrock x
        let nh4 = dfx.numh 1e-3 rosenbrock x
        let fxCorrect = rosenbrock x
        let hCorrect = rosenbrockHessian x
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(hCorrect, h1)
        Assert.CheckEqual(hCorrect, h2)
        Assert.CheckEqual(hCorrect, h3)
        Assert.CheckEqual(hCorrect, h4)
        Assert.True(hCorrect.allclose(nh1, 0.1))
        Assert.True(hCorrect.allclose(nh2, 0.1))
        Assert.True(hCorrect.allclose(nh3, 0.1))
        Assert.True(hCorrect.allclose(nh4, 0.1))

    [<Test>]
    member this.TestHessianNotTwiceDifferentiable () =
        let x = dfx.tensor([1.5, 2.5])
        let f (x:Tensor) = x.sum() // Not twice differentiable
        let fx1, h1 = dfx.fhessian f x
        let fx2, h2 = dfx.fh f x
        let h3 = dfx.hessian f x
        let h4 = dfx.h f x
        let nfx1, nh1 = dfx.numfhessian 1e-3 f x
        let nfx2, nh2 = dfx.numfh 1e-3 f x
        let nh3 = dfx.numhessian 1e-3 f x
        let nh4 = dfx.numh 1e-3 f x
        let fxCorrect = f x
        let hCorrect = dfx.zeros([2;2]) // Mathematically correct result for a function that is not twice differentiable, not achievable via autodiff
        Assert.CheckEqual(fxCorrect, fx1)
        Assert.CheckEqual(fxCorrect, nfx1)
        Assert.CheckEqual(fxCorrect, fx2)
        Assert.CheckEqual(fxCorrect, nfx2)
        Assert.CheckEqual(hCorrect, h1)
        Assert.CheckEqual(hCorrect, h2)
        Assert.CheckEqual(hCorrect, h3)
        Assert.CheckEqual(hCorrect, h4)
        Assert.True(hCorrect.allclose(nh1, 0.1))
        Assert.True(hCorrect.allclose(nh2, 0.1))
        Assert.True(hCorrect.allclose(nh3, 0.1))
        Assert.True(hCorrect.allclose(nh4, 0.1))

    [<Test>]
    member this.TestLaplacian () =
        let x = dfx.tensor([1.5, 2.5])
        let fx, l = dfx.flaplacian rosenbrock x
        let l2 = dfx.laplacian rosenbrock x
        let nfx, nl = dfx.numflaplacian 1e-3 rosenbrock x
        let nl2 = dfx.numlaplacian 1e-3 rosenbrock x
        let fxCorrect = rosenbrock x
        let lCorrect = (rosenbrockHessian x).trace()
        Assert.CheckEqual(fxCorrect, fx)
        Assert.CheckEqual(fxCorrect, nfx)
        Assert.CheckEqual(lCorrect, l)
        Assert.CheckEqual(lCorrect, l2)
        Assert.True(lCorrect.allclose(nl, 0.1))
        Assert.True(lCorrect.allclose(nl2, 0.1))

    [<Test>]
    member this.TestCurl () =
        let x = dfx.tensor([1.5, 2.5, 0.2])
        let fx, c = dfx.fcurl fvect3vect3 x
        let c2 = dfx.curl fvect3vect3 x
        let nfx, nc = dfx.numfcurl 1e-3 fvect3vect3 x
        let nc2 = dfx.numcurl 1e-3 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let cCorrect = dfx.tensor([-0.879814, -2.157828, 0.297245])
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(fxCorrect.allclose(nfx))
        Assert.True(cCorrect.allclose(c))
        Assert.True(cCorrect.allclose(c2))
        Assert.True(cCorrect.allclose(nc, 0.1))
        Assert.True(cCorrect.allclose(nc2, 0.1))

    [<Test>]
    member this.TestDivergence () =
        let x = dfx.tensor([1.5, 2.5, 0.2])
        let fx, d = dfx.fdivergence fvect3vect3 x
        let d2 = dfx.divergence fvect3vect3 x
        let nfx, nd = dfx.numfdivergence 1e-3 fvect3vect3 x
        let nd2 = dfx.numdivergence 1e-3 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let dCorrect = dfx.tensor(-0.695911)
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(fxCorrect.allclose(nfx))
        Assert.True(dCorrect.allclose(d))
        Assert.True(dCorrect.allclose(d2))
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))

    [<Test>]
    member this.TestCurlDivergence () =
        let x = dfx.tensor([1.5, 2.5, 0.2])
        let fx, c, d = dfx.fcurldivergence fvect3vect3 x
        let c2, d2 = dfx.curldivergence fvect3vect3 x
        let nfx, nc, nd = dfx.numfcurldivergence 1e-3 fvect3vect3 x
        let nc2, nd2 = dfx.numcurldivergence 1e-3 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let cCorrect = dfx.tensor([-0.879814, -2.157828, 0.297245])
        let dCorrect = dfx.tensor(-0.695911)
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(fxCorrect.allclose(nfx))
        Assert.True(cCorrect.allclose(c))
        Assert.True(cCorrect.allclose(c2))
        Assert.True(cCorrect.allclose(nc, 0.1))
        Assert.True(cCorrect.allclose(nc2, 0.1))
        Assert.True(dCorrect.allclose(d))
        Assert.True(dCorrect.allclose(d2))
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))


    [<Test>]
    member _.TestCanConfigure () =

        // Backup the current config before the test to restore in the end
        let configBefore = dfx.config()

        // Default reference backend with CPU
        let device = Device.Default
        dfx.config(device=Device.CPU)
        Assert.CheckEqual(Device.CPU, Device.Default)
        dfx.config(device=device)

        // Torch with default backend (CPU)
        let backend = Backend.Default
        dfx.config(backend=Backend.Torch)
        Assert.CheckEqual(Backend.Torch, Backend.Default)
        dfx.config(backend=backend)

        // Default reference backend with "int32"
        let dtype = Dtype.Default
        dfx.config(dtype=Dtype.Float64)
        Assert.CheckEqual(Dtype.Float64, Dtype.Default)
        dfx.config(dtype=dtype)

        // Restore the config before the test
        dfx.config(configBefore)

    [<Test>]
    member _.TestBackends () =
        let backends = dfx.backends() |> List.sort
        let backendsCorrect = [Backend.Reference; Backend.Torch] |> List.sort
        Assert.CheckEqual(backendsCorrect, backends)

    [<Test>]
    member _.TestDevices () =
        // Get devices for default reference backend
        let defaultReferenceBackendDevices = dfx.devices()
        Assert.CheckEqual([Device.CPU], defaultReferenceBackendDevices)

        // Get devices for explicitly specified reference backend
        let explicitReferenceBackendDevices = dfx.devices(backend=Backend.Reference)
        Assert.CheckEqual([Device.CPU], explicitReferenceBackendDevices)

        // Get CPU devices for explicitly specified reference backend
        let explicitReferenceBackendCPUDevices = dfx.devices(backend=Backend.Reference, deviceType=DeviceType.CPU)
        Assert.CheckEqual([Device.CPU], explicitReferenceBackendCPUDevices)

        // Get devices for explicitly specified Torch backend
        let explicitTorchBackendDevices = dfx.devices(backend=Backend.Torch)
        Assert.True(explicitTorchBackendDevices |> List.contains Device.CPU)
        let cudaAvailable = TorchSharp.torch.cuda.is_available()
        Assert.CheckEqual(cudaAvailable, (explicitTorchBackendDevices |> List.contains Device.GPU))

        let explicitTorchBackendDevices = dfx.devices(backend=Backend.Torch)
        Assert.True(explicitTorchBackendDevices |> List.contains Device.CPU)
        let cudaAvailable = TorchSharp.torch.cuda.is_available()
        Assert.CheckEqual(cudaAvailable, (explicitTorchBackendDevices |> List.contains Device.GPU))

    [<Test>]
    member _.TestIsBackendAvailable () =
        let referenceBackendAvailable = dfx.isBackendAvailable(Backend.Reference)
        Assert.True(referenceBackendAvailable)

    [<Test>]
    member _.TestIsDeviceAvailable () =
        let cpuAvailable = dfx.isDeviceAvailable(Device.CPU)
        Assert.True(cpuAvailable)

    [<Test>]
    member _.TestIsCudaAvailable () =
        let cudaAvailable = dfx.isCudaAvailable(Backend.Reference)
        Assert.False(cudaAvailable)

    [<Test>]
    member _.TestIsDeviceTypeAvailable () =
        Assert.True(dfx.isDeviceTypeAvailable(DeviceType.CPU))

        Assert.True(dfx.isDeviceTypeAvailable(DeviceType.CPU, Backend.Reference))
        Assert.False(dfx.isDeviceTypeAvailable(DeviceType.CUDA, Backend.Reference))

        Assert.True(dfx.isDeviceTypeAvailable(DeviceType.CPU, Backend.Torch))

        let cudaAvailable = TorchSharp.torch.cuda.is_available()
        let deviceSupported = dfx.isDeviceTypeAvailable(DeviceType.CUDA, Backend.Torch)
        Assert.CheckEqual(cudaAvailable, deviceSupported)

    [<Test>]
    member _.TestTensorAPIStyles () =
        let x = dfx.randn([5;5])

        // Base API
        dfx.seed(0)
        let y1 = x.dropout(0.2).leakyRelu(0.1).sum(1)

        // PyTorch-like API
        dfx.seed(0)
        let y2 = dfx.sum(dfx.leakyRelu(dfx.dropout(x, 0.2), 0.1), 1)

        // Compositional API for pipelining Tensor -> Tensor functions (optional, accessed through Dfx.Compose)
        dfx.seed(0)
        let y3 = x |> dfx.dropout 0.2 |> dfx.leakyRelu 0.1 |> dfx.sum 1

        Assert.CheckEqual(y1, y2)
        Assert.CheckEqual(y1, y3)

    [<Test>]
    member _.TestReverseDiffInit () =
        // Reverse derivative is initialized to an empty tensor (data: [], shape: [|0|], dim: 1)
        let x = dfx.tensor(1.).reverseDiff()
        Assert.AreEqual(x.derivative.shape, [|0|])

        // After propagation reverse derivative is a scalar tensor (shape: [||], dim: 0])
        let y = x.exp()
        y.reverse()
        Assert.AreEqual(x.derivative.shape, [||])

    [<Test>]
    member _.TestLoadSaveGeneric() =
        // string
        let v1 = "Hello, world!"
        let f1 = System.IO.Path.GetTempFileName()
        dfx.save(v1, f1)
        let v1b = dfx.load(f1)
        Assert.CheckEqual(v1, v1b)

        // int
        let v2 = 128
        let f2 = System.IO.Path.GetTempFileName()
        dfx.save(v2, f2)
        let v2b = dfx.load(f2)
        Assert.CheckEqual(v2, v2b)

        // float
        let v3 = 3.14
        let f3 = System.IO.Path.GetTempFileName()
        dfx.save(v3, f3)
        let v3b = dfx.load(f3)
        Assert.CheckEqual(v3, v3b)

        // bool
        let v4 = true
        let f4 = System.IO.Path.GetTempFileName()
        dfx.save(v4, f4)
        let v4b = dfx.load(f4)
        Assert.CheckEqual(v4, v4b)

        // list
        let v5 = [1, 2, 3]
        let f5 = System.IO.Path.GetTempFileName()
        dfx.save(v5, f5)
        let v5b = dfx.load(f5)
        Assert.CheckEqual(v5, v5b)

        // tuple
        let v6 = (1, 2, 3)
        let f6 = System.IO.Path.GetTempFileName()
        dfx.save(v6, f6)
        let v6b = dfx.load(f6)
        Assert.CheckEqual(v6, v6b)

        // dict
        let v7 = [("a", 1), ("b", 2), ("c", 3)]
        let f7 = System.IO.Path.GetTempFileName()
        dfx.save(v7, f7)
        let v7b = dfx.load(f7)
        Assert.CheckEqual(v7, v7b)

        // tuple of dicts
        let v8 = ([("a", 1), ("b", 2), ("c", 3)], [("a", 1), ("b", 2), ("c", 3)])
        let f8 = System.IO.Path.GetTempFileName()
        dfx.save(v8, f8)
        let v8b = dfx.load(f8)
        Assert.CheckEqual(v8, v8b)

        // tensor
        let v9 = dfx.tensor([1, 2, 3])
        let f9 = System.IO.Path.GetTempFileName()
        dfx.save(v9, f9)
        let v9b = dfx.load(f9)
        Assert.CheckEqual(v9, v9b)

        // model
        let v10 = Linear(10, 10)
        let f10 = System.IO.Path.GetTempFileName()
        dfx.save(v10, f10)
        let v10b:Model = dfx.load(f10)
        Assert.CheckEqual(v10.parametersVector, v10b.parametersVector)

