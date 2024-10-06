import sympy as sym
import numpy as np
import cupy as cp
import timeit

from sympy import lambdify


def f_cupy(func, x):
    x_cp = cp.asarray(x)
    result = func(x_cp[:,0], x_cp[:,1], x_cp[:,2], x_cp[:,3], x_cp[:,4], x_cp[:,5])
    return cp.asnumpy(result)


if __name__ =='__main__':

    x1, x2, x3, x4, x5, x6 = sym.symbols("x1, x2, x3, x4, x5, x6")

    V = 9*x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    lie = -0.2*x1**2*x3 - 18.0*x1**2 + 8.0*x1*x2 - 1.8*x1*x5**2 + 0.2*x2**2*x6 - 2.0*x2**2 - 2.0*x3**2 - 2.0*x4**2 - 2.0*x5**2 - 2.0*x6**2

    V_numpy = lambdify((sym.symbols("x1"), sym.symbols("x2"), sym.symbols("x3"), sym.symbols("x4"), sym.symbols("x5"), sym.symbols("x6")), V, "numpy")
    lie_numpy = lambdify((sym.symbols("x1"), sym.symbols("x2"), sym.symbols("x3"), sym.symbols("x4"), sym.symbols("x5"), sym.symbols("x6")), lie, "numpy")

    dataset = np.random.randn(20000000, 6)

    start = timeit.default_timer()
    result_numpy_V = V_numpy(dataset[:,0], dataset[:,1], dataset[:,2], dataset[:,3], dataset[:,4], dataset[:,5])
    result_numpy_lie = lie_numpy(dataset[:,0], dataset[:,1], dataset[:,2], dataset[:,3], dataset[:,4], dataset[:,5])
    end = timeit.default_timer()

    print(f"Numpy time usage: {end - start} second")

    start = timeit.default_timer()
    result_numpy_V = f_cupy(V_numpy, dataset)
    result_numpy_lie = f_cupy(lie_numpy, dataset)
    end = timeit.default_timer()


    print(f"Cupy time usage: {end - start} second")


   