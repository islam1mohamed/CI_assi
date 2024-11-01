import numpy as np

# Given function
def f(x):
    return 3*(x[0]**4) + 3*(x[0]**2)*(x[1]**2) + (x[0]**2) + 2*(x[1]**4)

# Initial point
point_init = np.array([1.0, 1.0])

# Analytical derivatives for verification
def f_dx1(x):
    return 12*x[0]**3 + 6*x[0]*x[1]**2 + 2*x[0]

def f_dx2(x):
    return 6*x[0]**2 * x[1] + 8 * x[1]**3

# Numerical derivative function using central difference
def numerical_derivative(func, x, h=1e-5):
    # Check if x is a single variable (number)
    if np.isscalar(x):
        # Use central difference formula for single-variable function
        return (func(x + h) - func(x - h)) / (2 * h)
    else:
        # Multi-variable case
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_forward = np.copy(x)
            x_backward = np.copy(x)
            x_forward[i] += h
            x_backward[i] -= h
            grad[i] = (func(x_forward) - func(x_backward)) / (2 * h)
        return grad

def mag(df):
    return np.sqrt(df[0]**2 + df[1]**2)

# variables
x_old = point_init
epsilon = 0.01
etta = 0.1
magnitude = 100000

while magnitude > epsilon:
    
    df = numerical_derivative(f, x_old)
    x_old = x_old - etta * df
    magnitude = mag(df)
    
print (f"the value using constant step {x_old} with magnitude df {magnitude}")

# ===========================
# NRGD
# ============================

# variables
x_old = point_init
epsilon = 0.01
etta = 0.1
magnitude_NR = 100000

# Numerical second derivative to compute the Hessian
def numerical_hessian(func, x, h=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_ij_plus = np.copy(x)
            x_ij_minus = np.copy(x)
            x_i_plus_j_minus = np.copy(x)
            x_i_minus_j_plus = np.copy(x)

            # Increment the indices i and j
            x_ij_plus[i] += h
            x_ij_plus[j] += h
            
            x_ij_minus[i] -= h
            x_ij_minus[j] -= h
            
            if i == j:
                # Diagonal elements: Second derivative with respect to x_i twice
                hessian[i, j] = (func(x_ij_plus) - 2 * func(x) + func(x_ij_minus)) / (h**2)
            else:
                # Off-diagonal elements: Mixed partial derivatives
                x_i_plus_j_minus[i] += h
                x_i_plus_j_minus[j] -= h
                x_i_minus_j_plus[i] -= h
                x_i_minus_j_plus[j] += h

                hessian[i, j] = (func(x_ij_plus) - func(x_i_plus_j_minus) - func(x_i_minus_j_plus) + func(x_ij_minus)) / (4 * h**2)

    return hessian

# Variables
point_init = np.array([1.0, 1.0])


while magnitude_NR > epsilon:

    df = numerical_derivative(f, x_old)
    Hess = numerical_hessian(f,x_old)

    
    x_old = x_old - np.dot(np.linalg.inv(Hess),df)    
    magnitude_NR = mag(df)
    

print (f" using NRGD final point is {x_old} at magnitude {magnitude_NR} ")

#=========================
# Steepest descent
#=========================
etta = 0.1
x_old = np.array([1.0,1.0])
df = numerical_derivative(f,x_old)

def x_new(etta):
    return x_old - etta * df

def phi(etta):
    return f(x_new(etta))

def numerical_second_derivative (func , x , h= 1e-5):
     return (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)

iter = 1
while mag(df) > epsilon:
    
    df = numerical_derivative(f,x_old)
    magnitude_SD = 1000

    while magnitude_SD > epsilon:
        
        dphi = numerical_derivative(phi , etta)
        d2phi = numerical_second_derivative(phi , etta)
        etta = etta - dphi/d2phi
        print(dphi)
        magnitude_SD = abs (dphi)
    x_old = x_new(etta)
    print (f"for the {iter} iteration we got etta : {etta}, with derivative magnitude :{dphi} , giving point {x_new(etta)} ")
    iter +=1
    
print (f" using steepest descent final point is {x_old} at magnitude {mag(df)} ")
