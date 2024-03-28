import numpy as np

def constraint_1(x):
    c_val = 1.0 - x[:,0] - x[:,1];
    return c_val;


def constraint_2(x):
    c_val = 1.0 - x[:,0]**2 - x[:,1]**2;
    return c_val;
