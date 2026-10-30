// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open Dfx
open Dfx.Util

// #nowarn "0058"

[<TestFixture>]
type TestDerivativesNested () =

    [<Test>]
    member _.TestDerivativesNestedPerturbationConfusion () =
        // 2nd order (fwd-on-fwd)
        // Perturbation confusion example
        // Siskind, J.M., Pearlmutter, B.A. Nesting forward-mode AD in a functional framework. Higher-Order Symb Comput 21, 361–376 (2008). https://doi.org/10.1007/s10990-008-9037-1
        // Page 4

        let x0 = dfx.tensor(1.)
        let y0 = dfx.tensor(2.)
        let d = dfx.diff (fun x -> x * dfx.diff (fun y -> x * y) y0) x0
        let dCorrect = dfx.tensor(2.)
        Assert.CheckEqual(dCorrect, d)


    [<Test>]
    member _.TestDerivativesNestedMin () =
        // 2nd order (fwd-on-fwd, because fgrad below is using fwd for scalar arguments) 
        // Siskind, J.M., Pearlmutter, B.A. Nesting forward-mode AD in a functional framework. Higher-Order Symb Comput 21, 361–376 (2008). https://doi.org/10.1007/s10990-008-9037-1
        // Page 2

        let rosenbrock (x:Tensor) (y:Tensor) = 
            (1. - x)**2 + 100. * (y - x**2)**2

        let x0 = dfx.tensor(0.9)

        let min f =
            let lr = 0.001
            let tolerance = 0.1
            let rec iter x =
                let fx, gx = dfx.fgrad f x
                if float fx <= tolerance then fx
                else iter (x - lr*gx)
            iter x0

        let fmin = min (fun x -> min (fun y -> rosenbrock x y))
        let fminCorrect = dfx.zero()

        Assert.True(fminCorrect.allclose(fmin, 0.1, 0.1))

    [<Test>]
    member _.TestDerivativesNestedChargedParticle () =
        // 3rd order (fwd-on-fwd-on-rev)
        // Nested optimization of a charged particle's trajectory
        // Siskind, J.M., Pearlmutter, B.A. Nesting forward-mode AD in a functional framework. Higher-Order Symb Comput 21, 361–376 (2008). https://doi.org/10.1007/s10990-008-9037-1
        // Page 13
        
        let dt = dfx.tensor(0.1)
        let x0 = dfx.tensor([0., 8.])
        let x'0 = dfx.tensor([0.75, 0.])
        let errorTolerance = 0.1

        let norm (x:Tensor) = x.pow(2).sum().sqrt()

        let naiveEuler (w:Tensor) = 
            let p x = (1 / norm (x - dfx.tensor([dfx.tensor(10.) + w*0; 10.-w])))
                    + (1 / norm (x - dfx.tensor([10, 0])))

            let mutable x = x0
            let mutable x' = x'0
            while float x[1] > 0. do
                let x'' = -dfx.grad p x
                x <- x + dt * x'
                x' <- x' + dt * x''

            let dtf = (0-x[1]) /x'[1]
            let xtf = x+dtf*x'
            xtf[0]*xtf[0]

        let argminNewton f x =
            let rec iter i x =
                let dfdx = dfx.diff f x
                if float (dfdx.abs()) < errorTolerance then i, x
                else iter (i+1) (x - dfdx/(dfx.diff (dfx.diff f) x))
            iter 1 x

        let w0 = dfx.tensor(0.)
        let i, wf = argminNewton naiveEuler w0
        let iCorrect, wfCorrect = 4, dfx.tensor(-0.266521)
        
        Assert.AreEqual(iCorrect, i)
        Assert.True(wfCorrect.allclose(wf, 0.01))

    [<Test>]
    member _.TestDerivativesNestedHessian () =
        // 2nd order (rev-on-fwd and rev-on-rev)
        // Compares Hessian-vector product to vector-Hessian product
        
        let rosenbrock (x:Tensor) = 
            let x, y = x[0], x[1]
            (1. - x)**2 + 100. * (y - x**2)**2

        // Analytical Hessian for Rosenbrock
        let rosenbrockHessian (x:Tensor) = 
            let x, y = x[0], x[1]
            dfx.tensor([[2.+1200.*x*x-400.*y, -400.*x],[-400.*x, 200.*dfx.one()]])

        // Jacobian-vector product (fwd)
        let jacobianv f x v =
            let _, d = dfx.evalForwardDiff f x v
            d

        // Vector-jacobian product (rev)
        let vjacobian f x v =
            let _, r = dfx.evalReverseDiff f x
            r v

        // Hessian-vector product (rev-on-fwd)
        let hessianv f x v =
            let gv xx = jacobianv f xx v
            let hv = vjacobian gv x (dfx.tensor(1.))
            hv

        // Vector-Hessian product (rev-on-rev)
        let vhessian f x v =
            let vg xx = vjacobian f xx (dfx.tensor(1.))
            let vh = vjacobian vg x v
            vh

        let x = dfx.randn(2)
        let v = dfx.randn(2)

        // Should be the same because Hessian is symmetric
        let hv = hessianv rosenbrock x v
        let vh = vhessian rosenbrock x v
        let hvCorrect = rosenbrockHessian(x).matmul(v)

        Assert.True(hvCorrect.allclose(hv, 0.01))
        Assert.True(hvCorrect.allclose(vh, 0.01))

    [<Test>]
    member _.TestDerivativesNestedTanh () =
        // 6th order (fwd-on-fwd-on-fwd-on-fwd-on-fwd-on-fwd)
        // Inspired by https://github.com/HIPS/autograd

        let sech (x:Tensor) = 1 / x.cosh()

        let f (x:Tensor) = x.tanh()
        let f' (x:Tensor) = let s = sech x in s*s
        let f'' (x:Tensor) = let s = sech x in -2.*x.tanh()*s*s
        let f''' (x:Tensor) = let t, s = x.tanh(), sech x in 4*t*t*s*s - 2*s*s*s*s
        let f'''' (x:Tensor) = let t, s = x.tanh(), sech x in 16*t*s*s*s*s - 8*t*t*t*s*s
        let f''''' (x:Tensor) = let t, s = x.tanh(), sech x in 8*s*s*(2*t*t*t*t + 2*s*s*s*s - 11*t*t*s*s)
        let f'''''' (x:Tensor) = let t, s = x.tanh(), sech x in -16*t*s*s*(2*t*t*t*t + 17*s*s*s*s - 26*t*t*s*s)

        let x = dfx.randn(1).squeeze()

        let d = dfx.diff
        let f'x = d f x
        let f''x = d (d f) x
        let f'''x = d (d (d f)) x
        let f''''x = d (d (d (d f))) x
        let f'''''x = d (d (d (d (d f)))) x
        let f''''''x = d (d (d (d (d (d f))))) x

        let f'xCorrect = f' x
        let f''xCorrect = f'' x
        let f'''xCorrect = f''' x
        let f''''xCorrect = f'''' x
        let f'''''xCorrect = f''''' x
        let f''''''xCorrect = f'''''' x

        Assert.True(f'xCorrect.allclose(f'x, 0.01))
        Assert.True(f''xCorrect.allclose(f''x, 0.01))
        Assert.True(f'''xCorrect.allclose(f'''x, 0.01))
        Assert.True(f''''xCorrect.allclose(f''''x, 0.01))
        Assert.True(f'''''xCorrect.allclose(f'''''x, 0.01))
        Assert.True(f''''''xCorrect.allclose(f''''''x, 0.01))

    [<Test>]
    member _.TestDerivativesOfConstantFunctions () =
        // A function that does not depend on its input has zero derivatives, in forward and reverse mode
        for combo in Combos.FloatingPointExcept16s do
            let xs = combo.tensor(2.)
            let xv = combo.tensor([1.; 2.])
            Assert.CheckEqual(combo.tensor(0.), dfx.diff (fun _ -> combo.tensor(3.)) xs)
            Assert.CheckEqual(combo.tensor(0.), dfx.diffn 2 (fun _ -> combo.tensor(3.)) xs)
            Assert.CheckEqual(combo.zeros([2]), dfx.grad (fun _ -> combo.tensor(3.)) xv)
            Assert.CheckEqual(combo.zeros([2]), dfx.jacobianv (fun _ -> combo.tensor([3.; 4.])) xv (combo.tensor([1.; 1.])))
            Assert.CheckEqual(combo.zeros([2; 2]), dfx.jacobian (fun _ -> combo.tensor([3.; 4.])) xv)
            Assert.CheckEqual(combo.zeros([2; 2]), dfx.hessian (fun _ -> combo.tensor(3.)) xv)

    [<Test>]
    member _.TestDerivativesNestedIndependentOfInnerVariable () =
        // The inner derivative of a function that depends on the outer variable but not on the inner one is zero.
        // It must not pick up the derivative at the outer nesting level, so d/dx (x + d/dy x^2) = 1.
        for combo in Combos.FloatingPointExcept16s do
            let x = combo.tensor(2.)
            let y = combo.tensor(3.)
            let xv = combo.tensor([2.])
            let yv = combo.tensor([3.])
            // fwd-on-fwd
            Assert.CheckEqual(combo.tensor(1.), dfx.diff (fun x -> x + dfx.diff (fun _ -> x * x) y) x)
            // rev-on-rev
            Assert.CheckEqual(combo.tensor([1.]), dfx.grad (fun x -> (x + dfx.grad (fun _ -> (x * x).sum()) yv).sum()) xv)
            // fwd-on-rev
            Assert.CheckEqual(combo.tensor(1.), dfx.diff (fun x -> x + (dfx.grad (fun _ -> x * x) yv).sum()) x)
            // rev-on-fwd
            Assert.CheckEqual(combo.tensor([1.]), dfx.grad (fun x -> (x + dfx.diff (fun _ -> (x * x).sum()) y).sum()) xv)

    [<Test>]
    member _.TestDerivativesDiffnVanishing () =
        // Higher derivatives that become zero before the requested order
        for combo in Combos.FloatingPointExcept16s do
            let x = combo.tensor(2.)
            Assert.CheckEqual(combo.tensor(0.), dfx.diffn 2 (fun x -> x) x)
            Assert.CheckEqual(combo.tensor(0.), dfx.diffn 3 (fun x -> x * x) x)
            Assert.CheckEqual(combo.tensor(2.), dfx.diffn 2 (fun x -> x * x) x)
            Assert.CheckEqual(combo.tensor(6.), dfx.diffn 3 (fun x -> x * x * x) x)


// Random pipelines of fwd-fwd-rev-fwd, etc.