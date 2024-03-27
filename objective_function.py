import numpy as np

def obj_func(x):
    f_val = 0.0;
    f_val = (1.0-x[0])**2 + 100.0*(x[1] - x[0]**2)**2;
    return f_val;

def grad_obj_func(x):
    df_val = np.array([0.0,0.0]);
    df_val[0] = -2*(1-x[0]) - 400*x[0]*(x[1] - x[0]**2);
    df_val[1] = 200*(x[1] - x[0]**2);
    return df_val;

def hess_obj_func(x):
    d2f_val = np.array( [[0.0,0.0], 
                         [0.0,0.0]] );
    d2f_val[0][0] = 2.0 + 800.0*x[0];
    d2f_val[0][1] = -400.0*x[0];
    d2f_val[1][0] = -400.0*x[0];
    d2f_val[1][1] = 200;
    return d2f_val;
