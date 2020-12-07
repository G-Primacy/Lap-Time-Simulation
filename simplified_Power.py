# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:17:12 2020

@author: Shawn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"Powertrain Gearing Parameters"
gear_ratio=np.asarray([2.68, 2.05, 1.71, 1.5, 1.36, 1.23])                      #Reclass gear ratio list as numpy array
primary_ratio=1.60;                                          "Crank->trans ratio"                
final_ratio=2.41;                                         "Chain drive ratio"
tire_OD=0.508;                                               "Tire Outer Diameter (m)"
drive_ratio=final_ratio*primary_ratio                       #Overrall powertrain ratio
shift_fudge_factor=1.25     #Shift a percentage amount about peak torque of current gear





"Import Powertrain Data to Panda database Matrix"

powerdata = pd.read_excel('Import Data/Powertrain_Data.xlsx', sheet_name='K5 GSXR1000 + GT1749');  #Import Engine Dyno Data to Pandas Database
powerdatamatrix = powerdata.to_numpy();                                             #Convert panda database to numpy array

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

"2D arrays for TQ x SPEED for all gears"
current_gear=0
dyno_wheel_torque = np.asarray([[0]*dyno_rpm.size]*gear_ratio.size);  "2D arrays for TQ x ENG SPEED for all gears"        
dyno_wheel_speed = np.asarray([[0]*dyno_rpm.size]*gear_ratio.size); "2D arrays for WHEEL SPEED x ENG SPEED for all gears"      

Torque_polynomial_degree=3                                          #Torque-speed polynomial degree

dyno_peak_torque=[];                                                   #Declare empty peak torque list
speed_torque_poly=np.asarray([[0]*(Torque_polynomial_degree+1)]*gear_ratio.size,float);     #Array for polynomial coefficents, wheel speed-tq for all gears
speed_torque_function=[0]*gear_ratio.size                                                   #Array of np.poly1D functions , speed-tq for all gears

rpm_torque_poly=np.asarray([[0]*(Torque_polynomial_degree+1)]*gear_ratio.size,float)        #Array for polynomial coefficents, engine speed-tq for all gears
rpm_speed_poly=np.asarray([[0]*(Torque_polynomial_degree+1)]*gear_ratio.size,float)         #Array for polynomial coefficents, engine speed-wheel speed for all gears

"""ENGINE & TRANSMISSION SOLVER"""
while current_gear < np.size(gear_ratio):                                                                     #Powertrain Solver
    dyno_wheel_torque[current_gear]=dyno_torque*gear_ratio[current_gear]*drive_ratio*1.0;    "Compute and populate, Axle torque = engine torque * drive ratio"
    dyno_wheel_speed[current_gear]=(dyno_rpm/60.0)*(2*np.pi)/(gear_ratio[current_gear]*drive_ratio)*(tire_OD/2); "Compute and populate, Wheel speed = engine speed * drive ratio"
    
    speed_torque_poly[current_gear]=np.polyfit(                                                         #Generate polynomial coefficients to speed-tq curve
        (dyno_wheel_speed[current_gear]),
        (dyno_wheel_torque[current_gear]),
        Torque_polynomial_degree)
    
    speed_torque_function[current_gear]= np.poly1d(speed_torque_poly[current_gear])                    #Generate poly1d from polynomial coefficients
    
    rpm_torque_poly[current_gear]=np.polyfit(                                                       #Generate polynomial for rpm-tq
        dyno_rpm,dyno_wheel_torque[current_gear],Torque_polynomial_degree)
    rpm_speed_poly[current_gear]=np.polyfit(                                                        #Generate polynomial for rpm-speed
        dyno_rpm,dyno_wheel_speed[current_gear],Torque_polynomial_degree)
    
    
    """PLOTTING RAW DATA TQ vs U"""
    plt.plot(dyno_wheel_speed[current_gear],dyno_wheel_torque[current_gear], label=("Gear: ", current_gear+1));    "Plot simdyno results: wheel torque at wheel speed"
    
    current_gear+=1;                                                                "Increment gear"  
    print("Solving powertrain gear:", current_gear)

"""DISPLAYING RAW DATA TQ vs U"""
plt.title('WHEEL TORQUE VS. TANGENTIAL SPEED ')
plt.ylabel('Torque (NM)')                               
plt.xlabel('Wheel Tangential Speed (m/s)')
axes = plt.gca()
axes.set_ylim([0,2000])
plt.legend()
plt.show()


"""PLOTTING & DISPLAYING POLYNOMIAl TQ vs U"""
current_gear=0
while current_gear < 6:
    p = speed_torque_function[current_gear]
    x = dyno_wheel_speed[current_gear]
    y = p(x)
    plt.plot(x, y,label=("Gear: ", current_gear+1))
    current_gear+=1;                                                                "Increment gear"  
axes = plt.gca()
axes.set_ylim([0,2000])
plt.title('WHEEL TORQUE VS. TANGENTIAL SPEED (POLYFIT)')
plt.ylabel('Torque (NM)')                               #Display Speed-Torque Curves
plt.xlabel('Wheel Tangential Speed (m/s)')
plt.legend()
plt.show()

#***************************************************************************************************************#
#***************************************************************************************************************#
"SHIFT LOGIC SOLVER" 

"""Determines the shift logic. Attempt to remain in domain of peak torque"""
print("Solving: shift logic")
#***************************************************************************************************************#
#***************************************************************************************************************#
speed_torque_derived=[0]*gear_ratio.size                               #Derived speed torque f'n
tqroots=[0]*gear_ratio.size                                             #Roots of the derived f'n
shift_points=[0]*gear_ratio.size                                        #Shift points

current_gear=0
while current_gear < 6:
    speed_torque_derived[current_gear]  =np.polyder(speed_torque_function[current_gear],m=1)            #Derive Tq-speed polynomials dTQ/dU, m=degrees of derivation
    tqroots[current_gear]=np.roots(speed_torque_derived[current_gear])
    
    shift_points[current_gear]= tqroots[current_gear][0]*shift_fudge_factor         #append the 1st (0th index) root of the tqroots function to the shift points array
    current_gear+=1;   
    
"""PLOTTING POLYNOMIAl dTQ/dU vs U"""
current_gear=0
while current_gear < 6:
    p= speed_torque_derived[current_gear]
    x = dyno_wheel_speed[current_gear]
    y = p(x)
    plt.plot(x, y,label=("Gear: ", current_gear+1))
    current_gear+=1;                                                                "Increment gear"  

plt.grid(b=True, which='major', axis='both')
plt.title('dTQ/dU VS. TANGENTIAL SPEED (POLYFIT)')
plt.xlabel('Wheel Tangential Speed (m/s)')
plt.legend()
plt.show()

"""PIECEWISE FUNCTION RETURNS TQ BASED ON SHIFT POINTS"""
def Wheel_force_solver(u):
    "Computes wheel torque, engine speed, current gear"
    if (0<= u) & (u < shift_points[0]):                                                             #1st gear domain, traction limited
        gear=0                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[0]<= u) & (u < shift_points[1]):                                              #2nd gear domain 
        gear=1                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[1]<= u) & (u < shift_points[2]):     #3rd gear domain
        gear=2                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[2]<= u) & (u < shift_points[3]):     #4th gear domain
        gear=3                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[3]<= u) & (u < shift_points[4]):     #5th gear domain
        gear=4                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[4]<= u):                             #6th gear domain
        gear=5                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif u < 0:
        return print("ERROR, NEGATIVE VELOCITY")
    else:
        return print("ERROR, INVALID VELOCITY")

def Engine_RPM_solver (u):
    if (0<= u) & (u < shift_points[0]):                                                             #1st gear domain, traction limited
        gear=0                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[0]<= u) & (u < shift_points[1]):                                              #2nd gear domain 
        gear=1                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[1]<= u) & (u < shift_points[2]):     #3rd gear domain
        gear=2                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[2]<= u) & (u < shift_points[3]):     #4th gear domain
        gear=3                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[3]<= u) & (u < shift_points[4]):     #5th gear domain
        gear=4                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif (shift_points[4]<= u):                             #6th gear domain
        gear=5                                                                                      #current gear
        wheel_torque = speed_torque_function[gear](u)/(tire_OD/2)                                   #wheel torque
        engine_speed = u*(gear_ratio[gear]*drive_ratio)/(tire_OD/2)                                 #engine speed
        return [wheel_torque,engine_speed,gear]                                                     #1st gear torque,rpm,gear
    elif u < 0:
        return print("ERROR, NEGATIVE VELOCITY")
    else:
        return print("ERROR, INVALID VELOCITY")

    return engine_speed