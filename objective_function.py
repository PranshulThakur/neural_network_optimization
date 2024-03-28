import numpy as np

def obj_func(x):
    f_val = (1.0-x[:,0])**2 + 100.0*(x[:,1] - x[:,0]**2)**2;
    return f_val;
