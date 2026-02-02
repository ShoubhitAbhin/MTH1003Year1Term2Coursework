import matplotlib.pyplot as plt
import numpy as np
import math

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
plt.plot(xExact, yExact, label="Analytical (no drag)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Comparison of numerical and analytical solutions (no drag)")
plt.legend()
plt.grid(True)
plt.savefig("Question1.png")




#############
# QUESTION 2 
#############


# new parameters for question 2
T = 1.5 # final time 
dt = 0.001 # timestep 


# initial conditions

t = 0
x = 0
y = Hball
vx = V*math.cos(alpha)
vy = V*math.sin(alpha)

xListQuestion2 = [x]
yListQuestion2 = [y]


# Forward Euler scheme

while y > 0:
    v = math.sqrt(vx**2 + vy**2)

    ax = -(0.5*Cd*p*A/m)*v*vx
    ay = -g - (0.5*Cd*p*A/m)*v*vy

    x = x + dt*vx
    y = y + dt*vy

    vx = vx + dt*ax
    vy = vy + dt*ay

    xListQuestion2.append(x)
    yListQuestion2.append(y)

# Plot trajectory

plt.clf()
plt.figure()
plt.plot(xNum, yNum, label="Euler (no drag)")
plt.plot(xListQuestion2, yListQuestion2, label = "Euler (with drag)")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Projectile Motion')
plt.legend()
plt.grid(True)
plt.savefig("Question2.png")




#############
# QUESTION 3
#############

def midpointStep(state, h):
    x, y, vx, vy = state
    
    dxdt1, dydt1, ax1, ay1 = derivatives(state)
    
    xMid  = x  + 0.5 * h * dxdt1
    yMid  = y  + 0.5 * h * dydt1
    vxMid = vx + 0.5 * h * ax1
    vyMid = vy + 0.5 * h * ay1
    
    stateMid = (xMid, yMid, vxMid, vyMid)
    
    dxdt2, dydt2, ax2, ay2 = derivatives(stateMid)
    
    xNew  = x  + h * dxdt2
    yNew  = y  + h * dydt2
    vxNew = vx + h * ax2
    vyNew = vy + h * ay2
    
    return xNew, yNew, vxNew, vyNew

# help function
def get_position_at_time(step_method, h_val, T_target):
    vx0 = V * np.cos(alpha)
    vy0 = V * np.sin(alpha)
    state = (0, Hball, vx0, vy0) 
    
    t_curr = 0
    while t_curr < T_target:
        if t_curr + h_val > T_target:
            h_final = T_target - t_curr
            state = step_method(state, h_final)
            t_curr = T_target
            break
        
        state = step_method(state, h_val)
        t_curr += h_val
        
    return state[0], state[1]

# convergence
print("Starting Convergence Study...")

T_check = 1.0 
h_ref = 0.00001 

ref_x, ref_y = get_position_at_time(midpointStep, h_ref, T_check)
print(f"Reference Position at T={T_check}s: x={ref_x:.4f}, y={ref_y:.4f}")

h_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
errors_euler = []
errors_midpoint = []

for h_val in h_values:
    
    x_e, y_e = get_position_at_time(eulerStep, h_val, T_check)
    dist_e = math.sqrt((x_e - ref_x)**2 + (y_e - ref_y)**2)
    errors_euler.append(dist_e)
    
  
    x_m, y_m = get_position_at_time(midpointStep, h_val, T_check)
    dist_m = math.sqrt((x_m - ref_x)**2 + (y_m - ref_y)**2)
    errors_midpoint.append(dist_m)

# plot
# plt.figure(figsize=(8, 6))
plt.clf()
plt.loglog(h_values, errors_euler, '-o', label='Forward Euler (1st Order)')
plt.loglog(h_values, errors_midpoint, '-s', label='Midpoint (2nd Order)')


plt.loglog(h_values, [10 * h for h in h_values], '--k', alpha=0.3, label='Reference Slope 1')
plt.loglog(h_values, [10 * h**2 for h in h_values], ':k', alpha=0.3, label='Reference Slope 2')

plt.xlabel('Timestep size h (s)')
plt.ylabel('Absolute Error (m)')
plt.title('Convergence Study: Error vs Timestep')
# plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig("Question3.png")

# print
# plt.clf()
slope_e = np.polyfit(np.log(h_values), np.log(errors_euler), 1)[0]
slope_m = np.polyfit(np.log(h_values), np.log(errors_midpoint), 1)[0]
print(f"Euler Slope: {slope_e:.2f} (Should be approx 1.0)")
print(f"Midpoint Slope: {slope_m:.2f} (Should be approx 2.0)")


# extreme values analysis
print("\nRunning Extreme Values Analysis...")


h_extreme_values = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.00002]

errors_euler_ext = []
errors_midpoint_ext = []

for h_val in h_extreme_values:
    ex, ey = get_position_at_time(eulerStep, h_val, T_check)
    e_err = math.sqrt((ex - ref_x)**2 + (ey - ref_y)**2)
    errors_euler_ext.append(e_err)
    
    mx, my = get_position_at_time(midpointStep, h_val, T_check)
    m_err = math.sqrt((mx - ref_x)**2 + (my - ref_y)**2)
    errors_midpoint_ext.append(m_err)

# plt.figure(figsize=(8, 6))

plt.loglog(h_extreme_values, errors_euler_ext, '-o', color='blue', label='Euler (with extremes)')

plt.loglog(h_extreme_values, errors_midpoint_ext, '-s', color='orange', label='Midpoint (with extremes)')

# plt.xlabel('Timestep h (s)')
# plt.ylabel('Absolute Error (m)')
# plt.title('Convergence Study: Extreme Values Analysis')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig("test.png")
# plt.savefig("Question3ExtremeConvergence.png")

print("Extreme analysis complete. Check 'Question3_Extreme_Convergence.png'.")
print("Notice how the error might flatten or spike at the far left (tiny h) and far right (huge h).")



# ----
 
####imports####
import numpy as np
import matplotlib.pyplot as plt
# base values
N=21 #(group 21)
m= 0.061          # kg
g= 9.81           # m/s^2
C= 0.5
R= 1.2
A= 1.9e-3         # m^2
omega = 20  # rad/s (negative = topspin, positive = backspin)
Cl = 0.2     # lift coefficient
kM = 0.5 * R * Cl * A / m
print("kM", kM)
 
# int conditions
v0= 30 #m/s
alpha= 5* np.pi /180
x0, y0= 0.0, 0.8
vx0= v0 * np.cos(alpha)
vy0= v0 * np.sin(alpha)
 
#time
dt= 0.001
Nmax= 100000
 
#array allocation#
x = np.zeros(Nmax)
y = np.zeros(Nmax)
vx = np.zeros(Nmax)
vy = np.zeros(Nmax)
ax_mag = np.zeros(Nmax)
ay_mag = np.zeros(Nmax)
ax_total = np.zeros(Nmax)
ay_total = np.zeros(Nmax)
t = np.zeros(Nmax)
 
#set int values##
x[0], y[0] = x0, y0
vx[0], vy[0] = vx0, vy0
 
 
n = 0
while y[n] > 0 and n < Nmax - 1:
 
    v = np.sqrt(vx[n]**2 + vy[n]**2)
 
    ax = -(0.5*C*R*A/m) * v * vx[n]
    ay = -g -(0.5*C*R*A/m) * v * vy[n]
    ax_mag = -kM * omega * vy[n]
    ay_mag = kM * omega * vx[n]
    ax_total = ax + ax_mag
    ay_total = ay + ay_mag
 
    x[n+1]  = x[n]  + vx[n]*dt
    y[n+1]  = y[n]  + vy[n]*dt
    vx[n+1] = vx[n] + ax_total*dt
    vy[n+1]= vy[n] + ay_total*dt
  
    
    
    
    
    t[n+1]  = t[n]  + dt
 
    n += 1
 
#Trim unused array entries
x = x[:n+1]
y = y[:n+1]
 

plt.clf()
plt.plot(x, y)
plt.axhline(0.9, linestyle='--')   # net height
plt.axvline(11.9, linestyle='--')  # net position
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.savefig("bweiufgkr.png")
 
