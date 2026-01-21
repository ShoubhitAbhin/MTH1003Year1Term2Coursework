import math
import matplotlib.pyplot as plt
import numpy as np


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

# function defining the ode system
def derivatives(state):
    x, y, vx, vy = state

    speed = math.sqrt(vx**2 + vy**2)

    ax = -(Cd * p * A / (2 * m)) * speed * vx
    ay = -g - (Cd * p * A / (2 * m)) * speed * vy

    dxdt = vx
    dydt = vy

    return dxdt, dydt, ax, ay

# eulers step function 
def eulerStep(state, h):
    dxdt, dydt, dvxdt, dvydt = derivatives(state)

    x, y, vx, vy = state

    xNew  = x  + h * dxdt
    yNew  = y  + h * dydt
    vxNew = vx + h * dvxdt
    vyNew = vy + h * dvydt

    return xNew, yNew, vxNew, vyNew

# time loop
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
x_vals = [s[0] for s in trajectory]
y_vals = [s[1] for s in trajectory]

plt.plot(x_vals, y_vals)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Tennis ball trajectory with drag")
plt.grid(True)
plt.show()



