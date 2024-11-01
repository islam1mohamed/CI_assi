import numpy as np

# givens
def f(x):
   return (3*(x[0]**4) + 3 * (x[0]**2) * (x[1]**2)+(x[0]**2) + 2 * (x[1]**4))
point_init = np.array([1,1])

#Sol
# let
etta = 0.1
epsilon = 0.01
inf = 1000
#----------------------
# using constant etta
#----------------------
def f_dx1(x):
    return (12*x[0]**3 + 6 *x[0]*x[1]**2 + 2*x[1])
def f_dx2(x) :
    return 6 * x[0]**2 *x[1] +8 *x[1]**3

def mag(df):
    return np.sqrt(df[0]**2 + df[1]**2)
x_old = point_init

magnitude = inf
while magnitude > epsilon:
    
    df = np.array([f_dx1(x_old) , f_dx2(x_old)])
    x_old = x_old - etta * df
    magnitude = mag(df)
    
    # print (np.array2string(x_old) +" give magnitude: "+ str(magnitude))

print (f"the value using constant step {x_old} with magnitude df {magnitude}")



#-----------------------
#NRGD
#-----------------------
def df(x):
    return np.array([f_dx1(x) , f_dx2(x)])

def H(x):
    f_x1x1 = 36 * x[0] **2 +6* x[1] **2 + 2
    
    f_x1x2 = 12*x[0]*x[1] 
    
    f_x2x2 = 6 *x[0]**2 + 24 * x[1]**2
    return np.array ([[f_x1x1 , f_x1x2],[f_x1x2 , f_x2x2]])
    

# variables
point_init= np.array([1.0, 1.0])
magnitude_NR = inf
x_old = point_init

while magnitude_NR > epsilon:
# for i in range(10):
    x_old = x_old - np.dot(np.linalg.inv(H(x_old)),df(x_old))    
    magnitude_NR = mag(df(x_old))
    
    
print (f" using NRGD final point is {x_old} at magnitude {magnitude_NR} ")

#=========================
# Steepest descent
#=========================
# value functions
def  phi(etta):
    return 792032 *etta**4 -175072* etta**3 +15100* etta **2-596*etaa +9

def d1phi(etta):
    return 3168128 *etta **3 - 525216* etta**2 +30200 * etta - 596

def d2phi (etta):
    return 9504384 *etta **2 -1050432 * etta + 30200


# variables

etta_node = 0.1
point_init= np.array([1.0, 1.0])
magnitude_SD = inf
epsilon = 0.001
etta_old = etta_node 
x = [1 , 1]
#solution

while abs(magnitude_SD)> epsilon:
    
    etta_new = etta_old - d1phi(etta_old)/d2phi(etta_old)
    etta_old = etta_new
    magnitude_SD = d1phi(etta_old)
x = [x[0]-df(x)[0]*etta_old, x[1]-df(x)[1]*etta_old]
print(f"the value using steepest descent {etta_old} giving initial point {x}")

while mag(df(x)) > epsilon:
    x = [x[0]-df(x)[0]*etta_old, x[1]-df(x)[1]*etta_old]

print(f"applying graidiant descent with optimum etta giving point {x}")


