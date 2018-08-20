import numpy as np
import math as ma
#Gradient descent of mulitvariable logistic equation
# function to be optimized
# p = 1/(1+e^(-4.8661+0.0345age+0.0023wclass+7.93*10^-7fnlweight+0.3611))
#x[1]=age,x[2]=wclass,x[3]=fnlweight,x[4]=ed_num,x[5]=mstatus
#x[6]=relat,x[7]=job,x[8]=race,x[9]=gender,x[10]=cg,x[11]=cl,
#x[12]=hours,x[13]=origin
b0 = -4.8661
b1 = 0.0345
b2 = 0.0023
b3 = 0.0000007693
b4 = 0.3611
b5 = -0.8166
b6 = -0.2909
b7 = -0.0475
b8 = -0.0541
b9 = -0.4735
b10 = 1.6769
b11 = 1.1355
b12 = 0.0328
b13 = -0.0107
# Mixed Random search
def f(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13 = x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12]
    y = ma.exp(-1*(b0+b1*x1+b2*x2+b3*x3+b4*x4+b5*x5+b6*x6+b7*x7+b8*x8+b9*x9+b10*x10+b11*x11+b12*x12+b13*x13))
    z = 1/(1+y)
    return z
def derivative2(f, var, d=0.001):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13 = var[0],var[1],var[2],var[3],var[4],var[5],var[6],var[7],var[8],var[9],var[10],var[11],var[12]
    fx1 = (f([x1+d/2,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])-f([x1-d/2,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]))/d
    fx2 = (f([x1,x2+d/2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])-f([x1,x2-d/2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]))/d
    fx3 = (f([x1,x2,x3+d/2,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])-f([x1,x2,x3-d/2,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]))/d
    fx4 = (f([x1,x2,x3,x4+d/2,x5,x6,x7,x8,x9,x10,x11,x12,x13])-f([x1,x2,x3,x4-d/2,x5,x6,x7,x8,x9,x10,x11,x12,x13]))/d
    fx5 = (f([x1,x2,x3,x4,x5+d/2,x6,x7,x8,x9,x10,x11,x12,x13])-f([x1,x2,x3,x4,x5-d/2,x6,x7,x8,x9,x10,x11,x12,x13]))/d
    fx6 = (f([x1,x2,x3,x4,x5,x6+d/2,x7,x8,x9,x10,x11,x12,x13])-f([x1,x2,x3,x4,x5,x6-d/2,x7,x8,x9,x10,x11,x12,x13]))/d
    fx7 = (f([x1,x2,x3,x4,x5,x6,x7+d/2,x8,x9,x10,x11,x12,x13])-f([x1,x2,x3,x4,x5,x6,x7-d/2,x8,x9,x10,x11,x12,x13]))/d
    fx8 = (f([x1,x2,x3,x4,x5,x6,x7,x8+d/2,x9,x10,x11,x12,x13])-f([x1,x2,x3,x4,x5,x6,x7,x8-d/2,x9,x10,x11,x12,x13]))/d
    fx9 = (f([x1,x2,x3,x4,x5,x6,x7,x8,x9+d/2,x10,x11,x12,x13])-f([x1,x2,x3,x4,x5,x6,x7,x8,x9-d/2,x10,x11,x12,x13]))/d
    fx10 = (f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10+d/2,x11,x12,x13])-f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10-d/2,x11,x12,x13]))/d
    fx11 = (f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11+d/2,x12,x13])-f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11-d/2,x12,x13]))/d
    fx12 = (f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12+d/2,x13])-f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12-d/2,x13]))/d
    fx13 = (f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13+d/2])-f([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13-d/2]))/d
    return np.array([fx1,fx2,fx3,fx4,fx5,fx6,fx7,fx8,fx9,fx10,fx11,fx12,fx13])
def init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,x5min,x5max,\
         x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,\
         x11min,x11max,x12min,x12max,x13min,x13max):
        x1_0 = x1min+np.random.random()*(x1max-x1min)
        x2_0 = x2min+np.random.random()*(x2max-x2min)
        x3_0 = x3min+np.random.random()*(x3max-x3min)
        x4_0 = x4min+np.random.random()*(x4max-x4min)
        x5_0 = x5min+np.random.random()*(x5max-x5min)
        x6_0 = x6min+np.random.random()*(x6max-x6min)
        x7_0 = x7min+np.random.random()*(x7max-x7min)
        x8_0 = x8min+np.random.random()*(x8max-x8min)
        x9_0 = x9min+np.random.random()*(x9max-x9min)
        x10_0 = x10min+np.random.random()*(x10max-x10min)
        x11_0 = x11min+np.random.random()*(x11max-x11min)
        x12_0 = x12min+np.random.random()*(x12max-x12min)
        x13_0 = x13min+np.random.random()*(x13max-x13min)
        return[ x1_0,x2_0,x3_0,x4_0,x5_0,x6_0,x7_0,x8_0,x9_0,x10_0,x11_0,x12_0,x13_0 ]
x1min,x1max = 17,90
x2min,x2max = 1.,9.
x3min,x3max = 12285.,1484705.
x4min,x4max = 1.,16.
x5min,x5max = 1.,7.
x6min,x6max = 1.,6.
x7min,x7max = 1.,15.
x8min,x8max = 1.,5.
x9min,x9max = 1.,2.
x10min,x10max = 0.,1.
x11min,x11max = 0.,1.
x12min,x12max = 1.,99.
x13min,x13max = 1.,42.
def maximize_fix(f,x0, N=1000):
    x0 = init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,x5min,x5max,\
         x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,\
         x11min,x11max,x12min,x12max,x13min,x13max)
    x_now = x0
    converged = False
    x_hist = []
    x_hist.append(x_now)
    dx = 0.0001
    for i in range(N):
        df_now = derivative2(f, x_now) 
        x_next = x_now + df_now*dx
        if f(x_next)<f(x_now):
            converged = True
            break
        else:
            x_now = x_next
            x_hist.append(x_now)
    return converged, np.array(x_hist), f(x_now)
def minimize_fix(f,x0, N=1000):
    x0 = init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,x5min,x5max,\
         x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,\
         x11min,x11max,x12min,x12max,x13min,x13max)
    x_now = x0
    converged = False
    x_hist = []
    x_hist.append(x_now)
    dx = 0.0001
    for i in range(N):
        df_now = derivative2(f, x_now) 
        x_next = x_now - df_now*dx
        if f(x_next)>f(x_now):
            converged = True
            
            break
        else:
            x_now = x_next
            x_hist.append(x_now)

    return converged, np.array(x_hist), f(x_now)
# Random search
minf1 = minimize_fix(f,init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,x5min,x5max,\
         x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,\
         x11min,x11max,x12min,x12max,x13min,x13max))
minfa = minf1[2]
minfc = minf1[1]
minf = minfa
N=100000
for i in range(N): 
    minf2 = minimize_fix(f,init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,x5min,x5max,\
         x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,\
         x11min,x11max,x12min,x12max,x13min,x13max))
    minfb = minf2[2]
    [x1_0,x2_0,x3_0,x4_0,x5_0,x6_0,x7_0,x8_0,x9_0,x10_0,x11_0,x12_0,x13_0] = init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,\
            x5min,x5max,x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,x11min,x11max,x12min,x12max,x13min,x13max)
    converged, x_hist, f_min = minimize_fix(f, [x1_0,x2_0,x3_0,x4_0,x5_0,x6_0,x7_0,x8_0,x9_0,x10_0,x11_0,x12_0,x13_0] )
    minfd = minf2[1]
    if minf<minfb:
        minf = minf
        x_final = minfc
    else:
        minf = minfb
        x_final = minfd
print('results of f_min: ', minf, '  number of iterations:  ', len(x_hist))
print(x_final[0])
maxf1 = maximize_fix(f,init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,x5min,x5max,\
         x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,\
         x11min,x11max,x12min,x12max,x13min,x13max))
maxfa = maxf1[2]
maxfc = maxf1[1]
maxf = maxfa
for i in range(N): 
    maxf2 = maximize_fix(f, init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,x5min,x5max,\
         x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,\
         x11min,x11max,x12min,x12max,x13min,x13max) )
    maxfb = maxf2[2]
    maxfd = maxf2[1]
    x_final2 = maxfd
    [x1_0,x2_0,x3_0,x4_0,x5_0,x6_0,x7_0,x8_0,x9_0,x10_0,x11_0,x12_0,x13_0] = init(x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,\
            x5min,x5max,x6min,x6max,x7min,x7max,x8min,x8max,x9min,x9max,x10min,x10max,x11min,x11max,x12min,x12max,x13min,x13max)
    converged, x_hist, f_max = maximize_fix(f, [x1_0,x2_0,x3_0,x4_0,x5_0,x6_0,x7_0,x8_0,x9_0,x10_0,x11_0,x12_0,x13_0] )
   
    if maxf>maxfb:
        maxf = maxf
        x_final2 = maxfc
    else:
        maxf = maxfb
        x_final2 = maxfd
print('results of f_max: ', maxf, '  number of iterations:  ', len(x_hist))
print(x_final2[0])
