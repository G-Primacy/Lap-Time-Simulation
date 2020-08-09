# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:01:39 2020

@author: Shawn's PC

The purpose of this lap time simulation is to diversify my personal programming skills. This is a translation of my MATLAB based lap time simulation
that I wrote in university.

Lacks some nice MATLAB things like Simulink/Simscape for suspension kinematics, but I may write my own in python down the line.

IE. I don;t want to pay for MATLAB, LUL.

"""
"Library Imports"
import math
import numpy as np
from scipy import *
from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
"""
Coordinate Systems:
    XYZ (Longitudinal, lateral, Vertical) - Car reference
    X'Y'Z' (Longitudinal, Lateral, Vertical) - Tire Reference
    u,v,w : X,Y,Z respective velocities (dX/dt, dY/dt, dZ/dt) , v=ra
    Ax,Ay,Az: X,Y,Z respective accelerations (du/dt, dv/dt, dw,dt)
    r,p: yaw, roll velocities
    N: yaw moment
"""

"Global Environment Parameters (SI)"
g=9.81;         "Gravity (m/s**2)"
rho_air=1.2754;        "Density of Air (kg/m**3)"

"DRIVER FUDGE FACTORS"
factor_fudge_brake = 0.95   #Braking fudge factor. Driver does not brake to the limit. On average can hit this percentage of optimal braking
factor_fudge_steering = 0.95 #Steering fudge factor. Driver does not steer at the limit. On average can hit this percentage of optinal steady state cornering

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
gear_ratio_list=[2.68, 2.05, 1.71, 1.5, 1.36, 1.23];         "Gear ratio list"
gear_ratio=np.asarray(gear_ratio_list)                      #Reclass gear ratio list as numpy array
primary_ratio=1.60;                                          "Crank->trans ratio"                
final_ratio=2.41;                                         "Chain drive ratio"
tire_OD=0.508;                                               "Tire Outer Diameter (m)"
drive_ratio=final_ratio*primary_ratio                       #Overtraill powertrain ratio

"Brake system parameters"
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

"Declaring Initial Kinematic Properties"
u=0;
v=0;
V=np.sqrt(u**2+v**2)

"Tire Statics Properties"
F_zRR=0
F_zRL=0
F_zFR=0
F_zFL=0

#***************************************************************************************************************#
#***************************************************************************************************************#
"AERO PARAMETERS"
print("Solving: Aero Parameters")
#***************************************************************************************************************#
#***************************************************************************************************************#


Area_Reference=1526825.7799;             "Frontal Reference Area (mm)"
Area_Reference_SI=Area_Reference/10E5;             "Frontal Reference Area in SI"
C_L=3.5;                        "Coefficient of Lift"
C_D=0.9;
x_CoP = -91.5;                  "Center of ure location along X originating from CoG"
z_CoP = 21;                     "Center of pressure location along Y originating from CoG"
z_frontsquat = 0;
z_rearsquat = 0;

def downforce_solver(u):
    "Given an input velocity, computes and returns the downforce (N) for longitudinal velocity"
    downforce_local = 0.5*(rho_air)*(Area_Reference_SI)*C_L*u**2;    "Downforce (N) = 1/2*Fluid Density*Reference Area*Lift*Velocity^2"
    return downforce_local
def drag_solver(u):
    "Given an input velocity, computes and returns drag (N) for longitudinal velocity"
    drag_local=-0.5*(rho_air)*(Area_Reference_SI)*C_D*u**2;   "Downforce (N) = 1/2*Fluid Density*Reference Area*Lift*Velocity^2"
    return drag_local


""""PLOTTING SIMPLE LONGITUDINAL DOWNFORCE AND DRAG - ASSUMING RIGID BODY AERO - NO AEROELASTICITY"""
u_local=np.linspace(0,60)
Downforce_local=0.5*(rho_air)*(Area_Reference_SI)*C_L*u_local**2
Drag_local=0.5*(rho_air)*(Area_Reference_SI)*C_D*u_local**2

plt.plot(u_local,Downforce_local, label='Downforce (N)')
plt.plot(u_local,Drag_local, label='Drag (N)')
plt.ylabel('Magnitude ()')
plt.xlabel('Longitudinal Speed (m/s)')
plt.legend()
plt.show()

#***************************************************************************************************************#
#***************************************************************************************************************#
"PANDAS IMPORT: SPREADSHEET DATAS"
print("Solving: Pandas Imports")
#***************************************************************************************************************#
#***************************************************************************************************************#
(9)
"Import Track Data to Panda data Matrix"

trackdata = pd.read_excel('Track_Data.xlsx');
trackdatamatrix = trackdata.to_numpy();

"Import Powertrain Data to Panda database Matrix"

powerdata = pd.read_excel('Powertrain_Data.xlsx', sheet_name='K5 GSXR1000 + GT1749');  #Import Engine Dyno Data to Pandas Database
powerdatamatrix = powerdata.to_numpy();                                             #Convert panda database to numpy array

"Import tire data: AVON 180/550/13"
tiredata_180_550_13_avon_100kg = pd.read_excel(r"G:\My Drive\The Race Car Project\Tire Data\Club F3 (AVON)\STAB RIG RAW DATA (1).XLS", 2);  #Import Engine Dyno Data to Pandas Database
tiredata_180_550_13_avon_100kg_matrix = tiredata_180_550_13_avon_100kg.to_numpy()

tiredata_180_550_13_avon_200kg = pd.read_excel(r"G:\My Drive\The Race Car Project\Tire Data\Club F3 (AVON)\STAB RIG RAW DATA (1).XLS", 3);  #Import Engine Dyno Data to Pandas Database
tiredata_180_550_13_avon_200kg_matrix = tiredata_180_550_13_avon_200kg.to_numpy()

tiredata_180_550_13_avon_300kg = pd.read_excel(r"G:\My Drive\The Race Car Project\Tire Data\Club F3 (AVON)\STAB RIG RAW DATA (1).XLS", 4);  #Import Engine Dyno Data to Pandas Database
tiredata_180_550_13_avon_300kg_matrix = tiredata_180_550_13_avon_300kg.to_numpy()

#***************************************************************************************************************#
#***************************************************************************************************************#
"TIRE DATA SOLVER"
print("Solving: Tires")
#***************************************************************************************************************#
#***************************************************************************************************************#
mu_long=1.60


camber_cases=5                                                            #camber data set quantity
load_cases=3                                                                #load data set quantity
i=0
j=0
slip_angles_local=[[0]*28]*camber_cases                                            #Declare slip angles matrix for 22 descrete slip angles, 4 cambers
lateral_force_local=[[0]*28]*camber_cases                                          #Declare lat forces matrix for 22 descrete lat forces, 4 cambers


"Single vertical load, CF vs SA at camber sweeped"
while(i != camber_cases):
    slip_angles_local=tiredata_180_550_13_avon_100kg_matrix[2:31,0]      #Append excel slip angle data to slip angle matrix
    lateral_force_local[i]=tiredata_180_550_13_avon_100kg_matrix[2:31,j+2]*1000    #Append excel force data to lat force matrix
    
    plt.plot(slip_angles_local,lateral_force_local[i], label=('Camber: ', i))

    j+=2                                                                #Load increment    
    i+=1                                                                #Incrememnt camber cases
plt.title('CF vs SA, Camber Sweep')
plt.ylabel('Force (N)')
plt.xlabel('Slip angle (deg)')
plt.legend()
plt.show()

"CF vs SLIP ANGLES, sweep for FY"
i=0
j=0
lateral_force=[[0]*28]*load_cases

lateral_force[i]=tiredata_180_550_13_avon_100kg_matrix[2:31,2]*1000    #Append excel force data to lat force matrix
lateral_force[i+1]=tiredata_180_550_13_avon_200kg_matrix[2:31,2]*1000    #Append excel force data to lat force matrix
lateral_force[i+2]=tiredata_180_550_13_avon_300kg_matrix[2:31,2]*1000    #Append excel force data to lat force matrix
plt.plot(slip_angles_local,lateral_force[i], label=('Load: 100kg'), linestyle='', marker='o')
plt.plot(slip_angles_local,lateral_force[i+1], label=('Load: 200kg'), linestyle='', marker='o')
plt.plot(slip_angles_local,lateral_force[i+2], label=('Load: 300kg'), linestyle='', marker='o')


plt.grid(b=True, which='major', axis='both')
plt.title('CF vs SA, 0 Camber, Mass Sweep')
plt.ylabel('Force (N)')
plt.xlabel('Slip angle (deg)')
plt.legend()
plt.show()

"CF (Gs) vs SLIP ANGLES, sweep for FY"
plt.plot(slip_angles_local,lateral_force[i]/100/g, label=('Load: 100kg'), linestyle='', marker='o')
plt.plot(slip_angles_local,lateral_force[i+1]/200/g, label=('Load: 200kg'), linestyle='', marker='o')
plt.plot(slip_angles_local,lateral_force[i+2]/300/g, label=('Load: 300kg'), linestyle='', marker='o')
                                                  #Incrememnt camber cases
plt.grid(b=True, which='major', axis='both')
plt.title('Accel (Gs)vs SA, 0 Camber, Mass Sweep')
axes = plt.gca()
axes.set_ylim([-2.5,2.5])
plt.ylabel('Acceleration (G) ')
plt.xlabel('Slip angle (deg)')
plt.legend()
plt.show()


#***************************************************************************************************************#
#***************************************************************************************************************#
"POWERTRAIN SOLVER" 
"""Import raw dyno data. Produce polynomial model of engine and transmission curves"""
print("Solving: Powertrain")
#***************************************************************************************************************#
#***************************************************************************************************************#

dyno_torque = powerdatamatrix[6,1:12];                  "Append Engine torque row to 1D list (Nm)"
dyno_rpm_list = powerdatamatrix[5,1:12];                      "Append Engine speed row to 1D list (rev/min)"
dyno_rpm= np.asarray(dyno_rpm_list,float)
dyno_power = dyno_rpm*dyno_torque/9.5488/1000;          "Compute Engine Power (KW)"

plt.plot(dyno_rpm,dyno_power, label='Engine Power (KW)');    "Plot simdyno results"
plt.plot(dyno_rpm,dyno_torque, label='Engine Torque (NM)');  "plot y: torque"
plt.title('GSXR1000 TURBO GT1749 DYNO')
plt.ylabel('Magnitude ()')
plt.xlabel('Engine Speed (rev/s)')
plt.legend()
plt.show()

"While Loop for wheel torque and speed"
current_gear=0
dyno_wheel_torque_list=[[0]*dyno_rpm.size]*gear_ratio.size;         "Declare empty 2D list for wheel torques at wheel speed of (rpm size columns)x(gear ratio size rows)"
dyno_wheel_torque = np.asarray(dyno_wheel_torque_list)              #Convert lists to 2D arrays
dyno_wheel_speed_list=[[0]*dyno_rpm.size]*gear_ratio.size;            "Declare empty 2D list  for wheel speeds"
dyno_wheel_speed = np.asarray(dyno_wheel_speed_list)

Torque_polynomial_degree=3                                          #Torque-speed polynomial degree

dyno_peak_torque=[];                                                   #Declare empty peak torque list
speed_torque_poly_list=[[0]*(Torque_polynomial_degree+1)]*gear_ratio.size;                        #Declare empty array of 6 degree polynomials for gears 1-6
speed_torque_poly=np.asarray(speed_torque_poly_list,float);            #Class speed-torque polynomial list to 
speed_torque_function=[0]*gear_ratio.size                              #Declare empty list for poly1d array for speed-torque equations at different gears

rpm_torque_list=[[0]*(Torque_polynomial_degree+1)]*gear_ratio.size                                #                rpm-tq list     
rpm_torque_poly=np.asarray(rpm_torque_list,float)                      #                rpm-tq array for polymial
rpm_speed_list=[[0]*(Torque_polynomial_degree+1)]*gear_ratio.size;                                #                rpm-u list
rpm_speed_poly=np.asarray(rpm_speed_list,float)                        #                rpm-u array for polynomial

"""ENGINE & TRANSMISSION SOLVER"""
while current_gear < 6:                                                                     #Powertrain Solver
    dyno_wheel_torque[current_gear]=dyno_torque*gear_ratio[current_gear]*drive_ratio*1.0;    "Compute Axle/Wheel Torque, parse  to wheel tq array"
    dyno_wheel_speed[current_gear]=(dyno_rpm/60.0)*(2*np.pi)/(gear_ratio[current_gear]*drive_ratio)*    (tire_OD/2); "Compute Wheel Speed, parse to wheel speed array"
    
    speed_torque_poly[current_gear]=np.polyfit(                                                         #Generate degree 8 polynomial coefficients to speed-tq curve
        (dyno_wheel_speed[current_gear]),
        (dyno_wheel_torque[current_gear]),
        Torque_polynomial_degree)
    
    speed_torque_function[current_gear]= np.poly1d(speed_torque_poly[current_gear])                    #Generate degree 8 polynomial function from polyfit coefficients
    
    rpm_torque_poly[current_gear]=np.polyfit(                                                       #Generate polynomial for rpm-tq
        dyno_rpm,dyno_wheel_torque[current_gear],Torque_polynomial_degree)
    rpm_speed_poly[current_gear]=np.polyfit(                                                        #Generate polynomial for rpm-speed
        dyno_rpm,dyno_wheel_speed[current_gear],Torque_polynomial_degree)
    plt.plot(dyno_wheel_speed[current_gear],dyno_wheel_torque[current_gear], label=current_gear+1);    "Plot simdyno results: wheel torque at wheel speed"
    
    current_gear+=1;                                                                "Increment gear"  
    print("Solving powertrain gear:", current_gear)
    
plt.title('WHEEL TORQUE VS. TANGENTIAL SPEED ')
plt.ylabel('Torque (NM)')                               #Display Speed-Torque Curves
plt.xlabel('Wheel Tangential Speed (m/s)')
axes = plt.gca()
axes.set_ylim([0,2000])
plt.legend()
plt.show()


"""PLOTTING POLYNOMIAl TQ vs U"""
current_gear=0
while current_gear < 6:
    p = speed_torque_function[current_gear]
    x = dyno_wheel_speed[current_gear]
    y = p(x)
    plt.plot(x, y)
    current_gear+=1;                                                                "Increment gear"  
axes = plt.gca()
axes.set_ylim([0,2000])
plt.title('WHEEL TORQUE VS. TANGENTIAL SPEED (POLYFIT)')
plt.ylabel('Torque (NM)')                               #Display Speed-Torque Curves
plt.xlabel('Wheel Tangential Speed (m/s)')
plt.show()



#***************************************************************************************************************#
#***************************************************************************************************************#
"SHIFT LOGIC SOLVER" 

"""Determines the shift logic. Attempt to remain in domain of peak torque"""
print("Solving: shift logic")
#***************************************************************************************************************#
#***************************************************************************************************************#
speed_torque_derived=[0]*gear_ratio.size                               #Derived speed torque f'n
tqroots=[0]*gear_ratio.size 
shift_points=[0]*gear_ratio.size 
shift_fudge_factor=1.05     #Shift a percentage amount before peak torque of the next gear

current_gear=0
while current_gear < 6:
    speed_torque_derived[current_gear]  =np.polyder(speed_torque_function[current_gear],m=1)            #Derive Tq-speed polynomials dTQ/dU
    tqroots[current_gear]=np.roots(speed_torque_derived[current_gear])
    
    shift_points[current_gear]= tqroots[current_gear][0]*shift_fudge_factor
    current_gear+=1;   
    
"""PLOTTING POLYNOMIAl dTQ/dU vs U"""
current_gear=0
while current_gear < 6:
    p= speed_torque_derived[current_gear]
    x = dyno_wheel_speed[current_gear]
    y = p(x)
    plt.plot(x, y)
    current_gear+=1;                                                                "Increment gear"  

plt.grid(b=True, which='major', axis='both')
plt.title('dTQ/dU VS. TANGENTIAL SPEED (POLYFIT)')
plt.xlabel('Wheel Tangential Speed (m/s)')
plt.show()

"We're going to shift at peak torque of the next gear cuz fuck it. Motorcycle gearing yo"
torque_solver=np.piecewise(u,
                           [u < shift_points[0], 
                            (shift_points[0]<= u) & (u < shift_points[1]), 
                            (shift_points[1]<= u) & (u < shift_points[2]), 
                            (shift_points[2]<= u) & (u < shift_points[3]), 
                            (shift_points[3]<= u) & (u < shift_points[4]), 
                            shift_points[4]<= u],
                           [lambda u: speed_torque_function[0](u), 
                             lambda u: speed_torque_function[1](u),
                             lambda u: speed_torque_function[2](u),
                             lambda u: speed_torque_function[3](u),
                             lambda u: speed_torque_function[4](u),
                             lambda u: speed_torque_function[5](u)]
                           )

"""PIECEWISE FUNCTION RETURNS TQ BASED ON SHIFT POINTS"""
def TQ_solver(u):
    "Compute traction limit domain"
    if (0<= u) & (u < shift_points[0]):                     #1st gear domain, traction limited
        return speed_torque_function[0](u)                  #1st gear torque
    elif (shift_points[0]<= u) & (u < shift_points[1]):     #2nd gear domain 
        return speed_torque_function[1](u)                  #2nd gear torque
    elif (shift_points[1]<= u) & (u < shift_points[2]):     #3rd gear domain
        return speed_torque_function[2](u)                  #3rd gear torque
    elif (shift_points[2]<= u) & (u < shift_points[3]):     #4th gear domain
        return speed_torque_function[3](u)                  #4th gear torque
    elif (shift_points[3]<= u) & (u < shift_points[4]):     #5th gear domain
        return speed_torque_function[4](u)                  #5th gear torque
    elif (shift_points[4]<= u):                             #6th gear domain
        return speed_torque_function[5](u)                  #6th gear torque
    elif u < 0:
        return print("ERROR, NEGATIVE VELOCITY")
    else:
        return print("ERROR, INVALID VELOCITY")
    

#***************************************************************************************************************#
#***************************************************************************************************************#
"SHIFT LOGIC SOLVER" 

"""Determines the shift logic. Attempt to remain in domain of peak torque"""
print("Defining brake distance solver")
#***************************************************************************************************************#
#***************************************************************************************************************#

def Brake_distance_solver(u1,u2):
    "Determine distance to decellerate to a desired velocity (u2) from (u1)."
    u_instantaneous=u1 #declare the instantaneous velocity
    t_step_local=0.01;  #deckare time step size
    print(u_instantaneous)
    distance_local=0;
    while u2<u_instantaneous:
        Force_y_local=weight_system + downforce_solver(u_instantaneous)     #Sum of forces Y
        Force_x_local=-mu_long*Force_y_local + drag_solver(u_instantaneous)         #Sum of forces X
        
        a_local= Force_x_local/m_S*factor_fudge_brake                               #Deceleration due of sum of forces X, 
        
        distance_local = distance_local + u_instantaneous*t_step_local
        u_instantaneous=u_instantaneous+t_step_local*a_local
              
    return distance_local

#***************************************************************************************************************#
#***************************************************************************************************************#
"STATIC TRACTIVE LIMITS"

"TBH, this is kind of useless. Static, steaty state assumptions. No drag"
#***************************************************************************************************************#
#***************************************************************************************************************#

"""Printing Wheel Force - Aero grip limit plot"""
current_gear=0
while current_gear < 6: 
    plt.plot(dyno_wheel_speed[current_gear],dyno_wheel_torque[current_gear]/(tire_OD/2), label=current_gear+1);    "Plot simdyno results: wheel Force at wheel speed"
    current_gear+=1

Downforce=0.5*(rho_air)*(Area_Reference_SI)*C_L*dyno_wheel_speed[5]**2    #Compute Downforce Curve
Longitudinal_limit_grip = mu_long*(weight_system+Downforce*0.5)*(1-r_FR)
plt.plot(dyno_wheel_speed[5],Longitudinal_limit_grip, label='Limit Grip (N)')  #Plot Longitudinal_limit_grip
plt.title('WHEEL FORCE & LONGITUDINAL LIMIT GRIP ')
plt.ylabel('Force (N)')                               #Display Speed-Torque Curves
plt.xlabel('Wheel Tangential Speed (m/s)')

axes = plt.gca()
axes.set_ylim([0,8000])
plt.legend()
plt.show()


#***************************************************************************************************************#
#***************************************************************************************************************#
"1D FWD LAP SOLVER"
#***************************************************************************************************************#
#***************************************************************************************************************#

""" 1D 0-100KMPH 2D Solver. Solid Suspension - no suspension transients"""

track_progress=0                    #Track sequence sequencer
i=0;                                "ith sequncer"
t_list=[0]*(trackdatamatrix.shape[0]);   "time elapsed per track segment"
t=np.asarray(t_list,float)
t_step=0.01;                            "time steps (s)"
linear_length=[];            "linear length of the ith track segment"

weight_transfer_longitudinal=0
wheel_force = speed_torque_function[0](u)/(tire_OD/2) #Current Potential Wheel Force
F_long_limit = mu_long*weight_system

while u<27:                                           #Solve while grip limited
    
    wheel_force = TQ_solver(u)/(tire_OD/2) #Current Potential Wheel Force
    Drag=-0.5*(rho_air)*(Area_Reference_SI)*C_D*u**2                   #Drag compute
    Downforce=0.5*(rho_air)*(Area_Reference_SI)*C_L*u**2              #Downforce compute  
                            
    Fy_local=weight_system*(1-r_FR) + weight_transfer_longitudinal + Downforce              #Sum of Forces in the Y Direction on the driven wheels (N)
    F_long_limit = mu_long*Fy_local                          #Compute Longitudinal acceleration limit (m*s**-2)
    
    if (wheel_force<F_long_limit)&(u<20):
        Fx_local=F_long_limit + Drag #Sum of Forces in the X Direction (N)
    else:
        Fx_local=wheel_force + Drag #Sum of Forces in the X Direction (N)
    
    a=np.abs(Fx_local/m_S)                                          #Acceleration is the limit grip - Drag
    weight_transfer_longitudinal = m_S*a*cgh/l                #Weight transfer solve
    
    u=u+a*t_step 
    t[0]=t[0]+t_step
    
    print("_____________________________")
    print("Drag:", Drag)
    print("Downforce:", Downforce)
    print("Fy,",Fy_local)
    print("Current Longitudinal Acceleration: ", a,"m/s**2")
    print("Current Velocity: ", u,"m/s")
    print("Current Wheel Force: ", wheel_force,"N")
    print("Current Force Limit: ", F_long_limit,"N")
    print("Long Weight Transfer: ", weight_transfer_longitudinal)
    print("Time Elapsed to 100km/h: ", t[0],"seconds")
    
    plt.plot(t[0],u)  #Plot u


while track_progress < trackdatamatrix.shape[0]:
    t[track_progress]+=t_step  
    track_progress+=1
    


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


""""2 Track, path simplified Solver 

10 DOF SOLVER: 
X,Y PATH CONSRTAINED, 
FREE: Body Z, XYZ YAW, 4 wheel Z, 2 Front wheel turn

Traverse predined path with highest accelerations: "The faster you go fast, the faster you are"

"""

"Program End Statements"


"""
root.mainloop();    "GUI Run Loop"
"""