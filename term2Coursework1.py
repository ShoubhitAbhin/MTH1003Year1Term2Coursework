
N = 21 # group number
m = (40+N) * 10e-3 # mass of tennis ball
g = 9.81 # gravitational acceleration
Cd = 0.5 # drag coefficient
p = 1.2 # density of air
A = (40-N) * 10e-4 # cross sectional area of the ball
Hball = 0.8 # initial height of the ball
L = 11.9 # distance from the baseline to the net
Hnet = 0.9 # net height
Fg = mg # force of gravity on the ball

v = 0 # velocity
absV = 0 # speed
alpha = 0 # angle

Fd = -0.5*Cd*p*A*absV*v # aerodynamic drag

