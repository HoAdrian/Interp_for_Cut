import numpy as np
import matplotlib.pyplot as plt

def newton_interp_poly(xi, yi):
  c = np.array(yi) #y_0...y_n
  n = len(yi)-1
  for k in range(1,n+1,1):
    d = xi[k] - xi[:k] #[x_k - x_0, ..., x_k - x_(k-1)]
    u = eval_newton_interp_poly(c[:k], xi[:k], xi[k])
    c[k] = (yi[k] - u)/np.prod(d)
  return c

def eval_newton_interp_poly(c, xi, x):
  '''
  xi are the interpolation nodes
  x is a point for evaluation
  c[i] is the leading coefficient of the poly interp x_0 ... x_i
  d[i] = x - xi[i]
  return the value of the interp poly at point x
  '''
  n = len(c)-1
  u = c[n]
  d = x - xi
  
  for k in range(n-1,-1,-1):
    u = u*d[k] + c[k]
  return u

if __name__ == "__main__":
  print("...Testing...")
  xi = np.array([-2, 0.5, 1, 2, 3]).astype(float)
  yi = np.array([-1, 3, 2, 4, 10]).astype(float)
  xs = np.linspace(-2, 3, 100)

  c = newton_interp_poly(xi, yi)

  ys = []
  for x in xs:
    y =  eval_newton_interp_poly(c,xi,x)
    ys.append(y)
  ys = np.array(ys)

  first_idx = 0
  plt.plot(xs[first_idx:], ys[first_idx:])
  plt.plot(xi[first_idx:], yi[first_idx:], ".")
  plt.show()

