# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:01:39 2020

@author: Shawn's PC

The purpose of this lap time simulation is to diversify my personal programming skills. This is a translation of my MATLAB based lap time simulation
that I wrote in university.

Lacks some nice MATLAB things like Simulink/Simscape for suspension kinematics, but I may write my own in python down the line.

IE. I don;t want to pay for MATLAB, LUL.

Coordinate Systems:
    XYZ (Longitudinal, lateral, Vertical) - Car reference
    X'Y'Z' (Longitudinal, Lateral, Vertical) - Tire Reference
    u,v,w : X,Y,Z respective velocities (dX/dt, dY/dt, dZ/dt) , v=ra
    Ax,Ay,Az: X,Y,Z respective accelerations (du/dt, dv/dt, dw,dt)
    r,p: yaw, roll velocities
    N: yaw moment
"""

"Library Imports"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simplified_Power as smp
import simplified_Aero as sma
#import Simplified_K as smk
#import simplified_Tire as smt

"Global Simulation Parameters"
time_step_limit=1000000       #Maximum number of time discrete steps
time_step_global=0.025       #global time step for time descritized lap sim

time=np.zeros(time_step_limit)                   #time list to track following parameters [0, 0.05, 0.10 ... , time_step_limit*time_step_global]
velocity_u=np.zeros(time_step_limit)               #longitudinal velocity in X direction list to track during simulation
velocity_y=np.zeros(time_step_limit)               #lateral velocity in Y direction list to track during simulation
velocity_w=np.zeros(time_step_limit)               #heaving velocity in Z direction list to track during simulation
acceleration_u=np.zeros(time_step_limit)           #longitudinal acceleration in X direction list~
acceleration_y=np.zeros(time_step_limit)           #lateral acceleration in Y direction ~ 
acceleration_w=np.zeros(time_step_limit)           #heave acceleration in Z~
yaw_rate=np.zeros(time_step_limit)                #yaw rate about Z~
roll_rate=np.zeros(time_step_limit)                #roll rate about X~
pitch_rate=np.zeros(time_step_limit)               #pitch rate about Y~
moment_yaw=np.zeros(time_step_limit)               #yawing moment about Z
moment_roll=np.zeros(time_step_limit)              #roll couple about X
moment_pitch=np.zeros(time_step_limit)             #pitching couple about Y

i=0
j=0
k=0

"Global Environment Parameters (SI)"
g=9.81;         "Gravity (m/s**2)"
rho_air=1.2754;        "Density of Air (kg/m**3)"

"DRIVER FUDGE FACTORS"
factor_fudge_brake = 0.95   #Braking fudge factor. Driver does not brake to the limit. On average can hit this percentage of optimal braking
factor_fudge_steering = 0.95 #Steering fudge factor. Driver does not steer at the limit. On average can hit this percentage of optinal steady state cornering
shift_fudge_factor=1.25     #Shift a percentage amount before peak torque of the next gear

"Vehicle Mass-Moment Parameters"
m_V= 340;       "vehicle mass"
m_D= 75;        "Driver mass"
m_S= m_V+m_D;   "System mass"
weight_system = m_S*g   #System Weight
r_FR = 0.42;    "front rear mass ratio"
r_LR = 0.5;     "left right mass ratio"
t_f = 1.250; "front track width"
t_r = 1.190; "rear track width"
cgh= 0.282; "Center of gravity height"
l = 2;       "Wheelbase"

"Powertrain Gearing Parameters"
tire_OD=0.508;                                               "Tire Outer Diameter (m)"

"In-wheel brake parameters"
area_piston_front=3.0*0.00064516;                      "Front Caliper piston area, in^2 -> m**2"
area_piston_rear=2.4*0.00064516;                       "Rear Caliper piston area, in^2 -> m**2"

Master_cylinder_area=np.pi*(19.05/2)**2;    "Master cylinder area (single)"
P_hydraulic_max=6.5e5;                      "User input max pressure threshhold warning"
mu_pad_avg=0.45                             #average coefficient of friction - willwood PolyMatrix E
height_pad_front=0.025                           
height_pad_rear=0.025                            
diameter_rotor_front=0.260                  #rotor diameter, front

"Brake pedal parameters"
from IPython.display import display, Image
display(Image(filename='G:\My Drive\The Race Car Project\Force Analysis\Pedal Box - Brake forces.png'))
"Pedal system force multiplication. Neglects pedal squish."
l_pedal= 0.18                   #length, from pedal pivot to middle of foot pad
l_mc=0.15                       #length, from master cylinder pivot to balance bar
l_ms=0.05                       #length, from pedal pivot to master cylinder pivot
l_of=0.135
pedal_angle = 85                #agnle, from horizontal plane, to pedal (deg)
pedal_angle_rad = pedal_angle*np.pi/180 #pedal angle, in (rad)
brake_bias=0.5                              #Perfentage of braking force delivered to the front brakes

diameter_MC_front= 0.75;                                "Font Master cylinder diameter (in)"
diameter_MC_rear= 0.75;                                 "Rear Master cylinder diameter (in)"
area_MC_front = np.pi*(diameter_MC_front/2)**2*0.00064516;   "Area, MC front (m**2)"
area_MC_rear = np.pi*(diameter_MC_rear/2)**2*0.00064516;     "Area, MC rear (m**2)"

"Declaring Initial Kinematic Properties"
u=0;
v=0;
V=np.sqrt(u**2+v**2)

#***************************************************************************************************************#
#***************************************************************************************************************#
"AERO PARAMETERS"
print("Solving: Aero Parameters")
#***************************************************************************************************************#
#***************************************************************************************************************#

Area_Reference=1526825.7799;             "Frontal Reference Area (mm)"
Area_Reference_SI=Area_Reference/10E5;             "Frontal Reference Area in SI"
C_L=3.1;                        "Coefficient of Lift"
C_D=0.9;
x_CoP = -91.5;                  "Center of ure location along X originating from CoG"
z_CoP = 21;                     "Center of pressure location along Y originating from CoG"
z_frontsquat = 0;
z_rearsquat = 0;

#***************************************************************************************************************#
#***************************************************************************************************************#
"PANDAS IMPORT: SPREADSHEET DATA"
print("Solving: Pandas Imports")
#***************************************************************************************************************#
#***************************************************************************************************************#
mu_long=1.60

trackdata = pd.read_excel('Import Data/Track_Data.xlsx',0,convert_float=True,skiprows=0);
trackdatamatrix = trackdata.to_numpy();

#***************************************************************************************************************#
def brake_pedal_force_solver_front(force_tractive):
    "Determines the required force applied to the pedal to achieve peak deceleration (friction limit) based"
    "Input: tire tractive force"
    torque_tire = force_tractive*tire_OD/2
    
    force_piston = torque_tire/(diameter_rotor_front/2-height_pad_front)/mu_pad_avg     #Force = (Torque/distance)/friction
    pressure_line=force_piston/area_piston_front                                        #Brake line pressure
    if pressure_line > 6.205*10**6:                                                     #Overpressure warning
        print("WARNING, PRESSURE: ",pressure_line)
    force_MC=pressure_line*area_MC_front                                                #Master cylinder force
    
    force_pedal=force_MC*l_of/l_pedal*np.cos(l_of/l_mc*(np.pi-pedal_angle_rad))         #FBD Solved relation between pedal force and force at th
    return force_pedal
    
#***************************************************************************************************************#
#***************************************************************************************************************#
"BRAKING LOGIC SOLVER" 

"""Determines the shift logic. Attempt to remain in domain of peak torque"""
print("Compiling brake state solvers")
#***************************************************************************************************************#
#***************************************************************************************************************#
def brake_distance_solver (u1,u2):
    "Computes distance to brake from u1 to u2"
    u_instantaneous=u1 #declare the instantaneous velocity
    t_step_local=0.01;  #deckare time step size
    distance=0;
    while u2<u_instantaneous:
        Force_y_local=weight_system + sma.downforce_solver(u_instantaneous)     #Sum of forces Y
        Force_x_local=-mu_long*Force_y_local + sma.drag_solver(u_instantaneous)         #Sum of forces X
        
        a_local= Force_x_local/m_S*factor_fudge_brake                               #Deceleration due of sum of forces X, '
    
        u_instantaneous=u_instantaneous+t_step_local*a_local           #recompute velocity
        distance= distance+ u_instantaneous*t_step_local              #sum braking distance
        
    
    return distance
def brake_state_solver(u1,u2,time_step,distance_remaining,driving_state):
    "Determine distance to decellerate to a desired velocity (u2) from (u1)."
    "Checks if projected brake distance is a percentage distance away from distance remaining"
    "Refines time step size to increase brake logic accuracy"
    "Initiate state change from state 0 to state 1"
    
    projected_distance= brake_distance_solver(u1,u2)                                            #project the distance to reach u2 from u1
    percent_difference = np.abs(projected_distance-distance_remaining)/distance_remaining       #check percentage difference between current distance left and projected brake distance
    print("Percentage difference in braking:",percent_difference)
    if percent_difference > 0.10:
        driving_state=0                  #Mainting driving state 0
        return [driving_state,time_step] #Maintain current time step size
    elif (percent_difference <= 0.10)&(percent_difference >= 0.05):
        driving_state=0                 #Mainting driving state 0
        time_step=time_step*0.95         #Refine time step by 100%
        return [driving_state,time_step]
    elif percent_difference <= 0.05:
        driving_state=1                 #driving state 1: initiate braking
        time_step = time_step_global    #return the time step size to the global size
        return [driving_state,time_step]

def print_function(time,u,v,w,A_x,A_y,A_z,yaw,roll,pitch,M_z,M_x,M_y):
    "Print Data"
    print("_____________________________")
    print("STATE: BRAKING FWD")
    print("Drag:", sma.drag_solver(u))
    print("Downforce:", sma.downforce_solver(u))
    print("Fy,",Fy_local)
    print("Current Longitudinal Acceleration: ", A_x*t_step ,"m/s**2")
    print("Current Velocity: ", u,"m/s")
    print("Time Elapsed to 100km/h: ", t,"seconds")
    print("Distance Elapsed: ", distance)
    print("Distance Remaining in Segment: ", length_remaining)
    print("Projected Braking Distance: ", brake_distance_solver(u,u_2))

class wheel_front_left:         #Tracks wheel properties
    def force_vertical(a):
        weight_transfer_longitudinal=0
        return weight_transfer_longitudinal
    def camber_solver():
        "Optimum K solved suspension parameters: "
        "Camber as function of vertical displacement"
        "Camber as function of roll (degrees)"
    def kpi_jacking_solver():
        "Computes front wheel kpi jacking force and dislacement"
    def IC_jacking_solver():
        "Computes vertical jacking force on body as function of lateral force"

class wheel_front_right:         #Tracks wheel properties
    def force_vertical(a):
        weight_transfer_longitudinal=0
        return weight_transfer_longitudinal
    def camber_solver():
        "Optimum K solved suspension parameters: "
        "Camber as function of vertical displacement"
        "Camber as function of roll (degrees)"
    def kpi_jacking_solver():
        "Computes front wheel kpi jacking force and dislacement"
    def IC_jacking_solver():
        "Computes vertical jacking force on body as function of lateral force"
    
class wheel_rear_left:         #Tracks wheel properties
    def force_vertical(a):
        weight_transfer_longitudinal=0
        return weight_transfer_longitudinal
    def camber_solver():
        "Optimum K solved suspension parameters: "
        "Camber as function of vertical displacement"
        "Camber as function of roll (degrees)"
    def kpi_jacking_solver():
        "Computes front wheel kpi jacking force and dislacement"
    def IC_jacking_solver():
        "Computes vertical jacking force on body as function of lateral force"

class wheel_rear_right:         #Tracks wheel properties
    def force_vertical(a):
        weight_transfer_longitudinal=0
        return weight_transfer_longitudinal
    def camber_solver():
        "Optimum K solved suspension parameters: "
        "Camber as function of vertical displacement"
        "Camber as function of roll (degrees)"
    def kpi_jacking_solver():
        "Computes front wheel kpi jacking force and dislacement"
    def IC_jacking_solver():
        "Computes vertical jacking force on body as function of lateral force"
    
class chassis_orientation:         #Tracks wheel properties
    def force_vertical(a):
        weight_transfer_longitudinal=0
        return weight_transfer_longitudinal
    def camber_solver():
        "Optimum K solved suspension parameters: "
        "Camber as function of vertical displacement"
        "Camber as function of roll (degrees)"
    def kpi_jacking_solver():
        "Computes front wheel kpi jacking force and dislacement"
    def IC_jacking_solver():
        "Computes vertical jacking force on body as function of lateral force"

#***************************************************************************************************************#

"TBH, this is kind of useless. Static, steaty state assumptions. No drag"
#***************************************************************************************************************#
#***************************************************************************************************************#

"""Printing Wheel Force - Aero grip limit plot"""
current_gear=0
while current_gear < 6: 
    p = smp.speed_torque_function[current_gear]/(tire_OD/2) 
    x = smp.dyno_wheel_speed[current_gear]
    y = p(x)
    plt.plot(x, y,label=("Gear: ", current_gear+1))
    current_gear+=1;                                                                "Increment gear"  

Downforce=0.5*(rho_air)*(Area_Reference_SI)*C_L*smp.dyno_wheel_speed[5]**2    #Compute Downforce Curve
Longitudinal_limit_grip = mu_long*(weight_system+Downforce*0.5)*(1-r_FR)
plt.plot(smp.dyno_wheel_speed[5],Longitudinal_limit_grip, label='Limit Grip (N)')  #Plot Longitudinal_limit_grip
plt.title('WHEEL FORCE & LONGITUDINAL LIMIT GRIP ')
plt.ylabel('Force (N)')                               #Display Speed-Torque Curves
plt.xlabel('Wheel Tangential Speed (m/s)')

axes = plt.gca()
axes.set_ylim([0,8000])
plt.legend()
plt.show()













#***************************************************************************************************************#
#***************************************************************************************************************#
""""
2 Track, path simplified Solver 

10 DOF SOLVER: 
X,Y PATH CONSRTAINED, 
FREE: Body Z, XYZ YAW, 4 wheel Z, 2 Front wheel turn

Traverse predined path with highest accelerations: "The faster you go fast, the faster you are"

5 States:
    Straight line accelleration     (state: 0)
    Straight line braking           (state: 1)
    Corner turn in                  (state: 2)
    Cornering                       (state: 3)
    Corner exit                     (state: 4)
    
    0:  6 DOF: Z Translation, Y Roll, 4 Wheel Z
        6 FIXED: XY Translation, XZ Yaw/Roll, 2 Wheel turn
        Solved Acceleration, Velocity
    1:  6 DOF: Z Translation, Y Roll, 4 Wheel Z
        6 FIXED: XY Translation, XZ Yaw/Roll, 2 Wheel turn
        Solved Acceleration, Velocity
    2:  10 DOF: Z Translation, Y Roll, 4 Wheel Z, XZ Yaw/Roll, 2 Wheel turn
        2 FIXED: XY Translation
        Free Acceleration, Solved Velocity
    3:  0 DOF: Pure Steady State
        12 FIXED: XY Translation, 2 Wheel turn, Body Z, XYZ YAW, 4 wheel Z, 2 Front wheel turn
        Solved Acceleration, Velocity
    4:  10 DOF: Z Translation, Y Roll, 4 Wheel Z, XZ Yaw/Roll, 2 Wheel turn
        2 FIXED: XY Translation
        Free Acceleration, Solved Velocity
"""
#***************************************************************************************************************#
#***************************************************************************************************************#
engine_speed=0
track_progress=0                    #Track sequence sequencer
i=0;                                "ith sequncer"
k=0;                                "kth sequencer"
j=0;                                "jth sequencer"                                  
t=0;                                "initial time"
distance=0;                                "initial displacement"
t_step=0.05;                            "time steps (s)"
linear_length=[];            "linear length of the ith track segment"

weight_transfer_longitudinal=0
wheel_force = smp.speed_torque_function[0](u)/(tire_OD/2) #Current Potential Wheel Force
F_long_limit = mu_long*weight_system
#***************************************************************************************************************#
"State 3 Solver"
#***************************************************************************************************************#

#***************************************************************************************************************#
"State 0 Solver"
#***************************************************************************************************************#
u_2=25
length_segment=trackdatamatrix[0,3]
state=0
while state==0:                                           #State 0
    
    wheel_force,engine_speed,current_gear = smp.Wheel_force_solver(u) #Current Potential Wheel Force
                            
    Fy_local=weight_system*(1-r_FR) + weight_transfer_longitudinal + sma.downforce_solver(u)              #Sum of Forces in the Y Direction on the driven wheels (N)
    F_long_limit = mu_long*Fy_local                          #Compute Longitudinal acceleration limit (m*s**-2)
    
    if (wheel_force<F_long_limit)&(u<20):
        Fx_local=F_long_limit + sma.drag_solver(u) #Sum of Forces in the X Direction (N)
    else:
        Fx_local=wheel_force + sma.drag_solver(u) #Sum of Forces in the X Direction (N)
    
    A_x=np.abs(Fx_local/m_S)                                          #Acceleration is the limit grip - Drag
    weight_transfer_longitudinal = m_S*A_x*cgh/l                #Weight transfer solve
    distance=distance+u*t_step;                                                    #compute distance traversed
    length_remaining=length_segment-distance
    u=u+A_x*t_step 
    t=t+t_step
    
    print("_____________________________")
    print("STATE: ACCEL FWD")
    print("Drag:", sma.drag_solver(u))
    print("Downforce:", sma.downforce_solver(u))
    print("Fy,",Fy_local)
    print("Current Longitudinal Acceleration: ", A_x*t_step ,"m/s**2")
    print("Current Velocity: ", u,"m/s")
    """print("Current Wheel Force: ", wheel_force,"N")
    print("Current Force Limit: ", F_long_limit,"N")
    print("Long Weight Transfer: ", weight_transfer_longitudinal)"""
    print("Time Elapsed: ", t,"seconds")
    print("Distance Elapsed: ", distance)
    print("Distance Remaining in Segment: ", length_remaining)
    print("Projected Braking Distance: ", brake_distance_solver(u,u_2))
    
    "Chechking if it is time to apply brakes"
    state,t_step=brake_state_solver(u,u_2,t_step,length_remaining,0)

    "Data Collection"
    time[i]=t                     #time list to track following parameters [0, 0.05, 0.10 ... , time_step_limit*time_step_global]
    velocity_u[i]=u               #longitudinal velocity in X direction list to track during simulation
    velocity_y[i]=v               #lateral velocity in Y direction list to track during simulation
    velocity_w[i]=0               #heaving velocity in Z direction list to track during simulation
    acceleration_u[i]=A_x           #longitudinal acceleration in X direction list~
    acceleration_y[i]=0           #lateral acceleration in Y direction ~ 
    acceleration_w[i]=0           #heave acceleration in Z~
    yaw_rate[i]=0                 #yaw rate about Z~
    roll_rate[i]=0                #roll rate about X~
    pitch_rate[i]=0               #pitch rate about Y~
    moment_yaw[i]=0               #yawing moment about Z
    moment_roll[i]=0              #roll couple about X
    moment_pitch[i]=m_S*A_x*cgh             #pitching couple about Y

    "Plot Data"
    i+=1
    """
while state==1:
    
    Fy_local=weight_system + downforce_solver(u)              #Sum of Forces in the Y Direction on the driven wheels (N)
    F_long_limit = mu_long*Fy_local                          #Compute Longitudinal acceleration limit (m*s**-2)

    Fx_local=(F_long_limit) + drag_solver(u) #Sum of Forces in the X Direction (N)
    
    A_x=-Fx_local/m_S                                          #Acceleration is the limit grip - Drag
    
    distance=distance+u*t_step;                                                    #compute distance traversed
    length_remaining=length_segment-distance
    u=u+A_x*t_step 
    t=t+t_step
    
    if u>u_2:
        state==1
    elif u<=u_2:
        state==2
        break
    
    print("_____________________________")
    print("STATE: BRAKING FWD")
    print("Drag:", drag_solver(u))
    print("Downforce:", downforce_solver(u))
    print("Fy,",Fy_local)
    print("Current Longitudinal Acceleration: ", A_x*t_step ,"m/s**2")
    print("Current Velocity: ", u,"m/s")
    print("Time Elapsed to 100km/h: ", t,"seconds")
    print("Distance Elapsed: ", distance)
    print("Distance Remaining in Segment: ", length_remaining)
    print("Projected Braking Distance: ", brake_distance_solver(u,u_2))
    
    "Data Collection"
    time[i]=t                     #time list to track following parameters [0, 0.05, 0.10 ... , time_step_limit*time_step_global]
    velocity_u[i]=u               #longitudinal velocity in X direction list to track during simulation
    velocity_y[i]=v               #lateral velocity in Y direction list to track during simulation
    velocity_w[i]=0               #heaving velocity in Z direction list to track during simulation
    acceleration_u[i]=A_x           #longitudinal acceleration in X direction list~
    acceleration_y[i]=0           #lateral acceleration in Y direction ~ 
    acceleration_w[i]=0           #heave acceleration in Z~
    yaw_rate[i]=0                 #yaw rate about Z~
    roll_rate[i]=0                #roll rate about X~
    pitch_rate[i]=0               #pitch rate about Y~
    moment_yaw[i]=0               #yawing moment about Z
    moment_roll[i]=0              #roll couple about X
    moment_pitch[i]=m_S*A_x*cgh             #pitching couple about Y

    "Plot Data"
    i+=1
 """

i-=1
plt.plot(time[0:i],velocity_u[0:i],label=('Speed'))
plt.title('Time vs. Velocity')
plt.ylabel('Magnitude')                               #Display Speed-Torque Curves
plt.xlabel('Time (s)')

plt.plot(time[0:i],acceleration_u[0:i],label=('Acceleration'))
plt.title('Time vs. Acceleration')
plt.ylabel('Magnitude')                                #Display Speed-Torque Curves
plt.xlabel('Time (s)')

plt.savefig('tessstttyyy.png', dpi=100)
plt.legend()
plt.show()

"""plt.title('Time vs speed graph')
plt.ylabel('Velocity (m/s)')                               #Display Speed-Torque Curves
plt.xlabel('Time (s)')"""

#***************************************************************************************************************#
#***************************************************************************************************************V
"VEHICLE DYNAMICS STATICS & STEADY STATE BEHAVIOUR"
#***************************************************************************************************************#
#***************************************************************************************************************#
"""REDECLARING, RECLASSING VARIABLES, SETTING INITIAL VALUES"""
l=                      2;              #Wheelbase                      (m)
l_a=                    l*(1-r_FR)      #CG location from front axle    (m)
l_b=                    l*r_FR          #CG kicatuin frim rear axle     (m) 
inertia_z=              261.7116 
radius_yaw_gyration=    1
F_L_front=              1
F_L_rear=               1
moment_yaw=             1
radius_path=            1
radius_inverted=        1
velocity_long=          1               #u, longitudinal velocity of the vehicle
velocity_lat=           1               #v, lateral velocity of thet vehicle
velocity_vehicle=       np.sqrt(u**2+v**2)           #V, absolute velocity of the vehicle
angle_steer=            1               #sigma_F, X axis of front wheel to vehicle centerline (rad)
angle_slip_F=           0               #alpha_F, Tire slip angle to V, front (rad)
angle_slip_R=           0               #alpha_R, Tire slip angle to V, rear (rad)
angle_vehicle=          1               #alpha_beta, Vehicle X axis angle to vector of V (rad)
stiffness_front=        1               #C_F, Wheel stiffness, front (N/rad)
stiffness_rear=         1               #C_R Wheel stiffness, rear (N/rad)
control_moment_gain=    stiffness_front*stiffness_rear/((stiffness_front + stiffness_rear)*weight_system) #C_0, control moment gain
angle_ack=              0               #sigma_a, ackermann reference steering angle (rad)
F_centrifugal=          0               #F_C, global vehicle centrifugal force(N)
ratio_steering=         0.21            #r_SW, Degrees of steer per degree of wheel ()

lateral_force_front=stiffness_front*angle_slip_F;
lateral_force_rear=stiffness_rear*angle_slip_R;

"""DERIVATIVES"""
Control_moment_derivative = -l_a*stiffness_front                                    #dN/dSigma (Moment-Steering)
yaw_damping_derivative = (1/V)*(l_a**2*stiffness_front + l_b**2*stiffness_rear)     #dN/dr (Moment-Yaw)
directional_stability_derivative = l_a*stiffness_front - l_b*stiffness_rear         #dN/dBeta (Moment-Slip angle)
 
control_force_derivative = -stiffness_front                                         #dY/dSigma (LatForce-Steering)
lateral_damping_derivative =stiffness_front + stiffness_rear                        #dY/dBeta (LatForce-Slip angle)
lateral_force_derivative = (1/V)*(l_a*stiffness_front - l_b*stiffness_rear)         #dY/dr (LatForce-Yaw)

factor_stability=       (m_S*directional_stability_derivative)\
    /(l*(directional_stability_derivative*control_force_derivative-lateral_damping_derivative*Control_moment_derivative))           #US/OS stability

"""ELEMENTAL VEHICLE DYNAMIC PROPERTIES: REF. RCVD 5.17: General Conclusions on Steady-State"""

neutral_steer_point = l/2;      #Neutral Steer Point, point where lateral forces produce no moments
static_margin=(neutral_steer_point-l_a)/l;      #Static Margin, difference between NSP and l_a normalized by wheelbase
critical_speed = np.sqrt(
    (directional_stability_derivative*(V*lateral_force_front)-
     lateral_damping_derivative*(V*yaw_damping_derivative))/(directional_stability_derivative*m_S)
    )
stability_index = 0;
gradient_understeer_d = weight_system*(
    directional_stability_derivative/(control_force_derivative*directional_stability_derivative-
                                      Control_moment_derivative*lateral_damping_derivative)
    )*57.3                      #understeer gradient, derivative form, deg/g deviation from Neutral Steer
gradient_understeer_SM = -57.3*(static_margin/control_moment_gain); #understeer gradient, SM form, deg/g deviation from Neutral Steer

"""CONTROL SYSTEM PROPERTIES: REF. RCVD 6."""
"General Equations"
#Natural Frequency = SQRT(STIFFNESS/MASS)
#Parallel Spring = (K_1 + K_2)dx
#Series Spring = inv(K_1**-1 + K_2**-1)dx
#Damping ratio = 1/2*(C/m*omega) = 1/2*(DAMPINGCOEFFICIENT/SQRT(STIFFNESS*MASS))
"General transient properties: 2DoF"
natural_frequency = np.sqrt(
    (stiffness_front*stiffness_rear*l**2)/(m_S**2*radius_yaw_gyration)
    *
    (1+factor_stability*V**2)/V**2
    )
#


