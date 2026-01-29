import math
import matplotlib.pyplot as plt
import numpy as np


#############
# QUESTION 1 
#############

N = 21 # group number
m = (40+N) * 1e-3 # mass of tennis ball
g = 9.81 # gravitational acceleration
Cd = 0.5 # drag coefficient
p = 1.2 # density of air
A = (40-N) * 1e-4 # cross sectional area of the ball

# initial conditions and checking values
Hball = 0.8 # initial height of the ball
L = 11.9 # distance from the baseline to the net
Hnet = 0.9 # net height

alpha = np.radians(30) # angle in radians - should be changed based on initial conditions
# alpha = np.degrees(45)
V = 20 # initial velocity - should be changed based on intial conditions

# variables that change with time
x = 0 # x component of the position vector
y = Hball # y component of the position vector
vxOft = V * np.cos(alpha) # x component of the velocity at time t
vyOft = V * np.sin(alpha) # y component of the velocity at time t
speed = np.sqrt(vxOft**2 + vyOft**2)

# time variables
t = 0 # intialises time 
h = 0.01 # timestep

# intial solution
"""
while y > 0:

    speed = math.sqrt(vxOft**2 + vyOft**2)

    ax = -(Cd * p * A / (2 * m)) * speed * vxOft
    ay = -g - (Cd * p * A / (2 * m)) * speed * vyOft

    vxOft = vxOft + h * ax
    vyOft = vyOft + h * ay

    x = x + h * vxOft
    y = y + h * vyOft

    t = t + h
"""


# -------------------

# function defining the ODE system
def derivatives(state):
    x, y, vx, vy = state

    speed = math.sqrt(vx**2 + vy**2)

    ax = -(Cd * p * A / (2 * m)) * speed * vx
    ay = -g - (Cd * p * A / (2 * m)) * speed * vy

    dxdt = vx
    dydt = vy

    return dxdt, dydt, ax, ay

# drag-free derivatives (for analytical comparison)
def derivativesNoDrag(state):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = 0
    dvydt = -g
    return dxdt, dydt, dvxdt, dvydt

# Eulers step function 
def eulerStep(state, h):
    dxdt, dydt, dvxdt, dvydt = derivatives(state)

    x, y, vx, vy = state

    xNew  = x  + h * dxdt
    yNew  = y  + h * dydt
    vxNew = vx + h * dvxdt
    vyNew = vy + h * dvydt

    return xNew, yNew, vxNew, vyNew

# Euler step without drag
def eulerStepNoDrag(state, h):
    dxdt, dydt, dvxdt, dvydt = derivativesNoDrag(state)
    x, y, vx, vy = state
    xNew  = x  + h * dxdt
    yNew  = y  + h * dydt
    vxNew = vx + h * dvxdt
    vyNew = vy + h * dvydt
    return xNew, yNew, vxNew, vyNew

# -------------------
# Q1: Drag-free numerical vs analytical solution

stateNoDrag = (0, Hball, V * np.cos(alpha), V * np.sin(alpha))
trajectoryNoDrag = []
timeVals = []

t = 0
while stateNoDrag[1] > 0:
    trajectoryNoDrag.append(stateNoDrag)
    timeVals.append(t)
    stateNoDrag = eulerStepNoDrag(stateNoDrag, h)
    t += h

xNum = np.array([s[0] for s in trajectoryNoDrag])
yNum = np.array([s[1] for s in trajectoryNoDrag])

# analytical solution
timeVals = np.array(timeVals)
xExact = V * np.cos(alpha) * timeVals
yExact = Hball + V * np.sin(alpha) * timeVals - 0.5 * g * timeVals**2

# drag trajectory calculation (unchanged)
state = (x, y, vxOft, vyOft)

while state[1] > 0:
    state = eulerStep(state, h)


# print final x and y
print("Final position: x =", state[0], "y =", state[1])

trajectory = []
state = (x, y, vxOft, vyOft)

while state[1] > 0:
    trajectory.append(state)
    state = eulerStep(state, h)

# convert to arrays
xVals = [s[0] for s in trajectory]
yVals = [s[1] for s in trajectory]

plt.figure()
plt.plot(xNum, yNum, label="Euler (no drag)")
plt.plot(xExact, yExact, '--', label="Analytical (no drag)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Comparison of numerical and analytical solutions (no drag)")
plt.legend()
plt.grid(True)
plt.show()



#############
# QUESTION 2 
#############
