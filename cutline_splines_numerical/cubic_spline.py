import numpy as np
import matplotlib.pyplot as plt


def cubic_spline(xi, yi):
    """
    Computes the cubic spline coefficients for a set of data points.

    Args:
        xi: A list of x-coordinates.
        yi: A list of y-coordinates.

    Returns:
        A tuple containing the x-coordinates, coefficients a, b, c, and d.
    """
    ### solve for ai, bi ci, di for i=0,...,n-1 ###
    ai = yi[:] # 0 ... n
    hi = np.diff(xi) # 0 ... n-1

    diag1 = np.diag(np.insert(hi[1:],0,0), k=1)
    diag0 = np.insert(2*(hi[0:-1]-hi[1:]),0,1)
    diag0 = np.insert(diag0,-1,1)
    diag0 = np.diag(diag0, k=0)
    diagNeg1 = np.diag(np.insert(hi[:-1], -1,0), k=-1)
    #print("diag0: ", diag0.shape)
    #print("diag1: ", diag1.shape)
    #print("diag-1: ", diagNeg1.shape)

    A = diag0 + diag1 + diagNeg1
    b = 3/hi[1:]*(ai[2:] - ai[1:-1]) - 3/hi[0:-1]*(ai[1:-1] - ai[0:-2])
    b = np.insert(b,0,0)
    b = np.insert(b,0,-1)

    # print("A", A.shape)
    # print("b", b.shape)

    ci = np.matmul(np.linalg.inv(A),b) #np.linalg.solve(A,b)
    di = 1/3*1/hi * (np.diff(ci))
    bi = 1/hi*(np.diff(ai)) - hi/3*(2*ci[:-1]+ci[1:])

    return ai[:-1], bi, ci[:-1], di

# interp: 0,1,2,...,n
# coeffs: 0,1,2,...,n-1
# list: 0,1,2,...,n
def eval_cubic_spline(xi,ai,bi,ci,di,eval_x):
    n = len(xi)
    print("num nodes: ", n)
    eval_y = []
    for i,x in enumerate(eval_x):
        if i %10==0:
            print(f"evaluating interp node {i}") 
        mask = xi>=x
        interval_idx = np.nonzero(mask)[0][0] - 1
        if interval_idx < 0:
            interval_idx+=1
        y = ai[interval_idx] + bi[interval_idx] * (x - xi[interval_idx]) + ci[interval_idx] * (x - xi[interval_idx]) ** 2 + di[interval_idx] * (x - xi[interval_idx]) ** 3
        eval_y.append(y)
    return eval_y   

if __name__ == "__main__":
  print("...Testing...")
  xi = np.array([-2, 0.5, 1, 2, 3])
  yi = np.array([-1, 3, 2, 4, 10])
  xs = np.linspace(-2, 3, 100)

  ai,bi,ci,di = cubic_spline(xi, yi)
  ys =  eval_cubic_spline(xi,ai,bi,ci,di,xs)

  first_idx = 0
  plt.plot(xs[first_idx:], ys[first_idx:])
  plt.plot(xi[first_idx:], yi[first_idx:], ".")
  plt.show()

