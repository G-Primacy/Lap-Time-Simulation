# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:01:18 2020

A program which digests raw tire data into:
Slip angle vs Lateral force curves & polynomials (swept for vertical load)
Friction ellipses
Spring rate and deflection
Pressure sensitivity levels

@author: Shawn
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"*********************************"
#Pandas Import: Tire Data, Raw
"*********************************"
Data_spring = ("G:/My Drive/The Race Car Project/Tire Data/British F3 (AVON)/F3 Front Rolling Springrate_0.XLS")
Data_friction = ("G:/My Drive/The Race Car Project/Tire Data/British F3 (AVON)/Front_8878_Stab_data.XLS")

Dataset_tires = pd.read_excel(Data_spring)    #Import entire raw tire data book as pandas dataframe

def sheet_quantity_solver (file_directory):
    xl = pd.ExcelFile(file_directory)                #read dataset
    sheet_quantity = len(xl.sheet_names)             #Number of sheets in dataset
    return sheet_quantity

def common_array_dimension_solver (file_directory):
    
    sheet_qty=sheet_quantity_solver(file_directory)
    
    i=0 
    sheet_array_sizes=np.zeros(shape=(sheet_qty,2),dtype=float)    #Declare array of size (sheet qty x 2)
    
    
    #Populates the sheet_array_sizes array with the shape of each sheet
    while i!=sheet_qty:
       sheet_temp = np.asarray(pd.read_excel(file_directory,sheet_name=i))  #assign sheet data to np array
       sheet_array_sizes[i]=np.shape(sheet_temp)                            #determine sheet data shape, assign to array sizes array
       i+=1
    i=0

    
    #Calculates the minumum array shape to collect all of the data within the raw dataset to np array
    sort_row_sizes= sorted(sheet_array_sizes[:,0])  #Sorts all row value, min -> max
    sort_col_sizes= sorted(sheet_array_sizes[:,1])  #Sorts all column values, min -> max
    
    max_row= sort_row_sizes[np.size(sort_row_sizes)-1]  #last (max) value in the row list
    max_col= sort_col_sizes[np.size(sort_col_sizes)-1]  #last (max) value in the column list
    
    array_dimension = [round(max_row),round(max_col)]                 #common largest array size, converted to integers
    return array_dimension

common_array_shape = common_array_dimension_solver(Data_spring)
sheet_quantity = sheet_quantity_solver(Data_spring)

Tire_data_shape = (sheet_quantity,common_array_shape[0],common_array_shape[1])
Tire_data_array = np.zeros(shape=Tire_data_shape)

def array_string_clean (array):
    i=0
    while i!=sheet_quantity:
        
        tire_data_temp = np.asarray(pd.read_excel(Data_spring,sheet_name=i))
        tire_data_temp_cleaned = [ x for x in tire_data_temp if x.isnumeric(9999999999) ]
        shape_temp = np.shape(tire_data_temp)
        Tire_data_array[i,0:shape_temp[0],0:shape_temp[1]]=tire_data_temp_cleaned
        i+=1
    return Tire_data_array


"**************************************************************************************"
#Spring Rate Data
"**************************************************************************************"

Dataset_tire_frequency_raw = np.asarray(pd.read_excel(Data_spring, sheet_name=2))    #Import entire raw tire data book as pandas dataframe
frequency_data_shape = [4,9]                                        #Dataset shape - MANUAL ENTRY
frequency_data_speeds = [0, 50/3.6, 70/3.6, 100/3.6, 170/3.6]       #Frequency Z axis: linear speed (m/s)
frequency_data_pressures = Dataset_tire_frequency_raw[8:12,0]       #Frequency Y axis: tire pressure (psi)
frequency_data_cambers=np.linspace(0,4,9)                           #Frequency X axis: slip angles (deg)
frequency_data=np.zeros(shape=[np.size(frequency_data_speeds),frequency_data_shape[0],frequency_data_shape[1]])     #frequency Data (3D)
i=0

#Populate the frequency data arrays
while (i!=np.size(frequency_data_speeds)):
    frequency_data[i] = Dataset_tire_frequency_raw[8+i*8:12+i*8,1:10]
    
    
    i+=1
"**************************************************************************************"
#Cornering Force Data
"**************************************************************************************"

Dataset_tire_friction_raw = np.asarray(pd.read_excel(Data_friction, sheet_name=3))    #Import entire raw tire data book as pandas dataframe
friction_data_shape = [14,17]                                      #Dataset shape - MANUAL ENTRY
friction_data_mass = np.asarray([Dataset_tire_friction_raw[2,1],Dataset_tire_friction_raw[19,1],Dataset_tire_friction_raw[36,1]])          #Frequency Z axis: mass load (kg)
friction_data_weight = np.asarray([Dataset_tire_friction_raw[2,1],Dataset_tire_friction_raw[19,1],Dataset_tire_friction_raw[36,1]])*9.81       #Frequency Z axis: force load (N)
friction_data_slip_angle = Dataset_tire_friction_raw[4:19,0]       #Frequency Y axis: tire pressure (psi)
friction_data_cambers=np.linspace(0,4,9)                           #Frequency X axis: slip angles (deg) MANUAL
                    #Dataset shape
frequency_data=np.zeros(shape=[np.size(friction_data_weight),friction_data_shape[0],friction_data_shape[1]])     #frequency Data (3D)
i=0
    
def find_nearest(array,value):
    #Find nearest index for non uniform axes. Must be sorted
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def round_down(num, divisor):
    #Rounds down num to the nearest divisor
    return num - (num%divisor)

def bilinear_interpolate(x_axis,y_axis,array,x,y):
    #Bilinear interpolation of values on an array using x and y independents. Requires uniform axes
    delta_x = x_axis[1]-x_axis[0]       #X axis iteration steps
    delta_y = y_axis[1]-y_axis[0]       #Y axis iteration steps
    
    x_round=round_down(x,delta_x)             #Round x to the nearest value in the x axis data
    y_round=round_down(y,delta_y)             #Round x to the nearest value in the x axis data
    
    #Find nearest index for uniform axes
    x_index = list(x_axis).index(x_round)       #we know that x lies in between x_index and x_index+1
    y_index = list(y_axis).index(y_round)       #we know that x lies in between y_index and y_index+1
    
    inter_block_A = array[y_index,x_index]*(x_axis[x_index+1]-x)*(y_axis[y_index+1]-y)/     ((x_axis[x_index+1]-x_axis[x_index])*(y_axis[y_index+1]-y_axis[y_index]))
    inter_block_B = array[y_index,x_index+1]*(x-x_axis[x_index])*(y_axis[y_index+1]-y)/     ((x_axis[x_index+1]-x_axis[x_index])*(y_axis[y_index+1]-y_axis[y_index]))
    inter_block_C = array[y_index+1,x_index]*(x_axis[x_index+1]-x)*(y-y_axis[y_index])/     ((x_axis[x_index+1]-x_axis[x_index])*(y_axis[y_index+1]-y_axis[y_index]))
    inter_block_D = array[y_index+1,x_index+1]*(x-x_axis[x_index])*(y-y_axis[y_index])/     ((x_axis[x_index+1]-x_axis[x_index])*(y_axis[y_index+1]-y_axis[y_index]))
    
    point= inter_block_A + inter_block_B + inter_block_C + inter_block_D
    return point
    
def trilinear_interpolate(x_axis,y_axis,z_axis,array,x,y,z):
    #Double bilinear interpolation between 2 sections on a z axis
    point_closest= find_nearest(z_axis,z)
    if z < point_closest:
        z_index=list(z_axis).index(point_closest)-1
    elif z > point_closest:
        z_index=list(z_axis).index(point_closest)
    else:
        print("ERROR")
    
    print(z_index)

   
    if z_index < (np.size(z_axis)-1):
         #Speed within the tire data: Trilinear interpolation
         point_interpolated_A = bilinear_interpolate(x_axis,y_axis,array[z_index],x,y)  #Bilinear interpolation of X and Y on lower bound Z
         point_interpolated_B = bilinear_interpolate(x_axis,y_axis,array[z_index+1],x,y) #Bilinear interpolation of X and Y on upper bound Z
         #Linear interpolation of A and B at Z
         point_interpolated= (point_interpolated_A)*(z_axis[z_index+1]-z)/(z_axis[z_index+1]-z_axis[z_index])+(point_interpolated_B)*(z-z_axis[z_index])/(z_axis[z_index+1]-z_axis[z_index])
    elif z_index == (np.size(z_axis)-1):
         #Maximum captured speed: Bilinear interpolation
         point_interpolated = bilinear_interpolate(x_axis,y_axis,array[z_index],x,y)
    else:
        print("ERROR, SPEED OUTSIDE CAPTURED DATA (Negative Velocity)")
    
    return point_interpolated

###################
"Friction Coefficients"
###################
#PAC2002, Steady state
LOAD_NOMINAL = 1060
#LONGITUDINAL_COEFFICIENTS
P_Cx1 =1.45    #Shape factor Cfx for longitudinal force
P_Dx1 =1.6314   #Longitudinal friction Mux at Fznom
P_Dx2 =-0.04906 #Variation of friction Mux with load
P_Dx3 = 0.006  #Variation of friction Mux with camber
P_Ex1 =0.4454   #Longitudinal curvature Efx at Fznom
P_Ex2 =0.2192   #Variation of curvature Efx with load
P_Ex3 =0.1125   #Variation of curvature Efx with load squared
P_Ex4 =0.1665   #Factor in curvature Efx while driving
P_Kx1 =43.63    #Longitudinal slip stiffness Kfx/Fz at Fznom
P_Kx2 =9.4735   #Variation of slip stiffness Kfx/Fz with load
P_Kx3 =0.023027     #Exponent in slip stiffness Kfx/Fz with load
P_Hx1 =-0.003839    #Horizontal shift Shx at Fznom
P_Hx2 =0.0044605    #Variation of shift Shx with load
P_Vx1 =0.04359      #Vertical shift Svx/Fz at Fznom
P_Vx2 =0.007515     #Variation of shift Svx/Fz with load
R_Bx1 =10   #Slope factor for combined slip Fx reduction
R_Bx2 =6    #Variation of slope Fx reduction with kappa
R_Cx1 =1    #Shape factor for combined slip Fx reduction
#LATERAL_COEFFICIENTS
P_Cy1 =0.7318 #Shape factor Cfy for lateral forces
P_Dy1 =1.6002 #Lateral friction Muy                             - PEAK FORCE ABSOLUTE
P_Dy2 =-0.3737 #Variation of friction Muy with load
P_Dy3 = -25  #Variation of friction Muy with camber             - HIGHER MAGNITUDE, HIGHER EFFECT
P_Ey1 =0.75 #Lateral curvature Efy at Fznom
P_Ey2 =4.1214 #Variation of curvature Efy with load             
P_Ey3 =-0.2125   #Variation of curvature Efy with load squared   - PEAKINESS
P_Ey4 =0.2665   #Factor in curvature Efy while driving
P_Ky1 =-69 #Maximum value of stiffness Kfy/Fznom                - STIFFNESS ABSOLUTE
P_Ky2 = 0.052  #Load at which Kfy reaches maximum value              -P_Ky2 < 1 stiffness reduces with load. P_Ky2 > 1 stiffness increases with load
P_Ky3 = 0.0451  #Variation of Kfy/Fznom with inclination
P_Hy1 =0.000 #Horizontal shift Shy at Fznom
P_Hy2 =0.000    #Variation of shift Shy with load
P_Hy3 =-0.0   #Variation of shift Shy with inclination
P_Vy1 =0.01 #Vertical shift in Svy/Fz at Fznom
P_Vy2 =-0.1 #Variation of shift Svy/Fz with load
P_Vy3 =2 #Variation of shift Svy/Fz with inclination
P_Vy4 =-0.32 #Variation of shift Svy/Fz with inclination and load
R_By1 =12 #Slope factor for combined Fy reduction
R_Cy1 =1 #Shape factor for combined Fy reduction
#ALIGNING_COEFFICIENTS
Q_Bz1 =5.58 #Trail slope factor for trail Bpt at Fznom
Q_Bz2 =-1.5 #Variation of slope Bpt with load
Q_Bz3 =-0.888 #Variation of slope Bpt with load squared
Q_Bz4 =-0.199 #Variation of slope Bpt with inclination
Q_Bz5 =0.321 #Variation of slope Bpt with load squared
Q_Bz9 =0.09 #Variation of slope Bpt with inclination
Q_Bz10=-0.255 #Slope factor Br of residual moment Mzr
Q_Cz1 =1.2 #Shape factor Cpt for pneumatic trail
Q_Dz1 =0.12 #Peak trail Dpt = Dpt*(Fz/Fznom*R0)
Q_Dz2 =-0.02 #Variation of peak Dpt with load
Q_Dz3 = 0.03 #Variation of peak Dpt with inclination
Q_Dz4 = 0.03#Variation of peak Dpt with inclination squared.
Q_Dz6 = 0.1   #Peak residual moment Dmr = Dmr/ (Fz*R0)
Q_Dz7 = 0.898   #Variation of peak factor Dmr with load
Q_Dz8 = 0.005   #Variation of peak factor Dmr with inclination
Q_Dz9 = 0.001   #Variation of Dmr with inclination and load
Q_Ez1 =-32.6 #Trail curvature Ept at Fznom
Q_Ez2 = -29.6 #Variation of curvature Ept with load
Q_Ez3 = -0.126313751   #Variation of curvature Ept with load squared
Q_Ez4 = 0.6356488674   #Variation of curvature Ept with sign of Alpha-t
Q_Ez5 = -2.6687967   #Variation of Ept with inclination and sign Alpha-t
Q_Hz1 =  0.0022895106   #Trail horizontal shift Sht at Fznom
Q_Hz2 = -0.000951929998   #Variation of shift Sht with load
Q_Hz3 = 0.031030   #Variation of shift Sht with inclination
Q_Hz4 =  0.05791840   #Variation of shift Sht with inclination and 
#LONGITUDINAL_SCALING_FACTORS
L_FZo=5  #Scale factor of nominal (rated) load
L_CZ=1   #Scale factor of vertical tire stiffness
L_CX=1   #Scale factor of Fx shape factor
L_MUX=1  #Scale factor of Fx peak friction coefficient
L_EX=1   #Scale factor of Fx curvature factor
L_KX=1   #Scale factor of Fx slip stiffness
L_HX=1   #Scale factor of Fx horizontal shift
L_VX=1   #Scale factor of Fx vertical shift
L_GAX=1  #Scale factor of inclination for Fx
L_CY=1   #Scale factor of Fy shape factor
L_MUY=1  #Scale factor of Fy peak friction coefficient
L_EY=1   #Scale factor of Fy curvature factor
L_KY=1   #Scale factor of Fy cornering stiffness
L_HY=1   #Scale factor of Fy horizontal shift
L_VY=1   #Scale factor of Fy vertical shift
L_GAY=1  #Scale factor of inclination for Fy
L_TR=1   #Scale factor of peak of pneumatic trail
L_RES=1  #Scale factor for offset of residual moment
L_GAZ=1  #Scale factor of inclination for Mz
L_MX=1   #Scale factor of overturning couple
L_VMX=1  #Scale factor of Mx vertical shift
L_MY=1   #Scale factor of rolling resistance moment
#LATERAL SCALING FACTORS
L_CZ=1   #Scale factor of vertical tire stiffness
L_GAX=1  #Scale factor of inclination for Fx
L_CY=1   #Scale factor of Fy shape factor
L_MUY=1  #Scale factor of Fy peak friction coefficient
L_EY=1   #Scale factor of Fy curvature factor
L_KY=1   #Scale factor of Fy cornering stiffness
L_HY=1   #Scale factor of Fy horizontal shift
L_VY=1   #Scale factor of Fy vertical shift
L_GAY=1  #Scale factor of inclination for Fy
L_TR=1   #Scale factor of peak of pneumatic trail
L_RES=1  #Scale factor for offset of residual moment
L_MX=1   #Scale factor of overturning couple
L_VMX=1  #Scale factor of Mx vertical shift
L_MY=1   #Scale factor of rolling resistance moment
L_VY=1      #Scale factor of Fy vertical shift
L_T=1
#Global lateral coefficients to be used in alignment
d_FZ = None
L_Y = None
S_Hy = None
S_Vy = None
#COEFFICIENTS
MU_y = None
C_y = None
D_y = None
E_y = None
K_y0 = None
K_y = None
B_y = None

SR=np.linspace(0,1,50)
SA=np.linspace(-7,7,50)
SA_rad=SA*np.pi/180
def MF_longitudinal_function (slip_ratio,load,camber,t_slip):
    #PAC2002, Steady state
    #TURN SLIP RATIO = ` DURING SLIP
    turn_slip=t_slip
    
    d_FZ =(load - (LOAD_NOMINAL*L_FZo))/(LOAD_NOMINAL*L_FZo)  #Vertical Load Increment
    S_Hx = (P_Hx1 + P_Hx2*d_FZ)*L_HX
    slip_ratio_x = slip_ratio + S_Hx
    L_x = camber*L_GAZ
    
    #MF COEFFICIENTS
    C_x = P_Cx1 *L_CX
    MU_x = (P_Dx1+P_Dx2*d_FZ)*(1-P_Dx3*camber**2)*L_MUX
    D_x = MU_x*load*turn_slip
    E_x = (P_Ex1 + P_Ex2*d_FZ+ P_Ex3*d_FZ**2)*(1-P_Ex4*np.sign(slip_ratio_x))*L_EX
    
    #Slip Stiffness
    K_x = load*(P_Kx1+P_Kx2*d_FZ)*np.exp(P_Kx3*d_FZ)*L_KX
    B_x = K_x/(C_x*D_x)
    S_Hx = (P_Hx1 + P_Hx2*d_FZ)*L_HX
    S_Vx = load*(P_Vx1+P_Vx2*d_FZ)*L_VX*L_MUX*turn_slip
    
    
    F_x0=D_x*np.sin(C_x*np.arctan(B_x*slip_ratio - E_x*(B_x*slip_ratio)))+S_Vx
    return F_x0

def MF_lateral_function (slip_angle,load,camber,t_slip):
    #2002 lateral force formula, Steady State


    #TURN SLIP RATIO = ` DURING SLIP
    turn_slip=t_slip
    
    d_FZ =(load - (LOAD_NOMINAL*L_FZo))/(LOAD_NOMINAL*L_FZo)  #Vertical Load Increment
    L_Y = camber*L_GAY
    
    global S_Hy
    S_Hy = (P_Hy1+P_Hy2*d_FZ)*L_HY + P_Hy3*L_Y
    global S_Vy 
    S_Vy = load*((P_Vy1+P_Vy2*d_FZ)*L_VY+(P_Vy3+P_Vy4*d_FZ)*L_Y)*L_MUY*t_slip
    global alpha_y 
    alpha_y = slip_angle + S_Hy
    G_y = load**1.1/load
    
    #COEFFICIENTS
    global MU_y
    MU_y= (P_Dy1+P_Dy2*d_FZ)*(1-P_Dy3*camber**2)*L_MUY
    global C_y 
    C_y = P_Cy1*L_CY
    global D_y 
    D_y = MU_y*load*turn_slip
    global E_y 
    E_y = (P_Ey1+P_Ey2*d_FZ)*(1-(P_Ey3+P_Ey4)*np.sign(alpha_y))*L_EY
    global K_y0 
    K_y0 = P_Ky1*load*np.sin(2*np.arctan(load/(P_Ky2*LOAD_NOMINAL*L_FZo)))*L_FZo*L_KY
    global K_y 
    K_y = K_y0*(1-P_Ky3*np.abs(L_Y))*turn_slip
    global B_y 
    B_y = K_y*(C_y/D_y)
    
    #Frequency Scaling
    
    #Camber Stiffness
    K_yi0=P_Hy3*K_y0+load*(P_Vy3+P_Vy4*d_FZ)
    

    F_y=D_y*np.sin(C_y*np.arctan(B_y*alpha_y-E_y*(B_y*alpha_y-np.arctan(B_y*alpha_y))))+S_Vy
    return (F_y)

def MF_alignment_function (slip_angle,load,camber,t_slip,radius):
    #2002 lateral force formula, Steady State
    #Inputs: Slip angle, load, camber/inclination angle
    L_R = 1
    d_FZ =(load - (LOAD_NOMINAL*L_FZo))/(LOAD_NOMINAL*L_FZo)  #Vertical Load Increment
    
    L_Z = camber*L_GAZ

    
    #Coefficients alignment
    D_t = load*(Q_Dz1+Q_Dz2*d_FZ)*(1+ Q_Dz3*L_Z+Q_Dz4*L_Z**2)*radius/(LOAD_NOMINAL)*L_T*t_slip
    C_t = Q_Cz1
    B_t = (Q_Bz1+Q_Bz2*d_FZ+Q_Bz3*d_FZ**2)*(1+Q_Bz4*L_Z+Q_Bz5*np.abs(L_Z))*L_KY/L_MUY
    E_t = (Q_Ez1+Q_Ez2*d_FZ+Q_Ez3*d_FZ**2)
    #Lateral force at contact patch
    F_y = MF_lateral_function(slip_angle,load,camber,t_slip)
    
    #Pnematic trail to contact patch
    S_Ht = Q_Hz1 + Q_Hz2*d_FZ +(Q_Hz3+Q_Hz4*d_FZ)*L_Z
    S_Hf = S_Vy/K_y
    alpha_t = slip_angle + S_Ht
    alpha_r = slip_angle + S_Hf
    t = D_t*np.cos(C_t*np.arctan(B_t*alpha_t-E_t*(B_t*alpha_t-np.arctan(B_t*alpha_t))))*np.cos(slip_angle)
    B_r = (Q_Bz9*L_KY/L_MUY+Q_Bz10*B_y*C_y)*t_slip
    C_r = 1 #TEMP
    D_r = load*((Q_Dz6 + Q_Dz7*d_FZ)*L_R+(Q_Dz8+ Q_Dz9*d_FZ)*L_Z)*radius*L_MUY+t_slip-1
    K_z = -t*K_y
    
    M_Zr = D_r*np.cos(C_r*np.arctan(B_r*alpha_r))*np.cos(slip_angle)
    print(t)
    print(F_y)
    print(M_Zr)
    
    
    M_z = -t*F_y/2 + M_Zr
    return M_z


plt.plot(SR,MF_longitudinal_function(SR, friction_data_weight[0], 0,1),label=("300kg"))
plt.plot(SR,MF_longitudinal_function(SR, friction_data_weight[1], 0,1),label=("400kg"))
plt.plot(SR,MF_longitudinal_function(SR, friction_data_weight[2], 0,1),label=("500kg"))

plt.ylabel('Force (N)')                               #Display Speed-Torque Curves
plt.xlabel('Slip angle (deg)')
plt.legend()
plt.show()



"400kg"
#plt.plot(friction_data_slip_angle,Dataset_tire_friction_raw[55:71,2]*1000, label="Measured Data, 600kg")
plt.plot(friction_data_slip_angle,Dataset_tire_friction_raw[4:19,2]*1000, label="Measured Data, 300kg")
plt.plot(SA,MF_lateral_function(SA_rad,friction_data_weight[0],0,1),label="MF Tire model, 300kg")
plt.ylabel('Force (N)')                               #Display Speed-Torque Curves
plt.xlabel('Slip angle (deg)')
axes = plt.gca()
axes.set_ylim([-8000,8000])

plt.legend()
plt.show()
"500kg"
plt.plot(friction_data_slip_angle,Dataset_tire_friction_raw[21:36,2]*1000, label="Measured Data, 400kg")
plt.plot(SA,MF_lateral_function(SA_rad,friction_data_weight[1],0,1),label="MF Tire model, 400kg")
plt.ylabel('Force (N)')                               #Display Speed-Torque Curves
plt.xlabel('Slip angle (deg)')
axes = plt.gca()
axes.set_ylim([-8000,8000])

plt.legend()
plt.show()
"600kg"
plt.plot(friction_data_slip_angle,Dataset_tire_friction_raw[38:53,2]*1000, label="Measured Data, 500kg")
plt.plot(SA,MF_lateral_function(SA_rad,friction_data_weight[2],0,1),label="MF Tire model, 500kg")
plt.ylabel('Force (N)')                               #Display Speed-Torque Curves
plt.xlabel('Slip angle (deg)')
axes = plt.gca()
axes.set_ylim([-8000,8000])

plt.legend()
plt.show()

"""plt.plot(SR,MF_longitudinal_function(SR, friction_data_weight[0], 3,1),label=("300kg, 3deg camber"))
plt.plot(SR,MF_longitudinal_function(SR, friction_data_weight[1], 3,1),label=("400kg, 3deg camber"))
plt.plot(SR,MF_longitudinal_function(SR, friction_data_weight[2], 3,1),label=("500kg, 3deg camber"))"""

#plt.plot(SA,MF_alignment_function(SA_rad, friction_data_weight[2], 3,1,0.5),label=("500kg, 3deg camber"))
#lt.legend()
#plt.show()

def MFL_Norm (slip_ratio,load,camber,t_slip):
    
    F_X = MF_longitudinal_function(slip_ratio, load, camber, t_slip)
    
    F_Xn = F_X/(load)
    
    return (F_Xn)

plt.plot(SR,MFL_Norm(SR, friction_data_weight[0], 0,1),label=("300kg"))
plt.plot(SR,MFL_Norm(SR, friction_data_weight[1], 0,1),label=("400kg"))
plt.plot(SR,MFL_Norm(SR, friction_data_weight[2], 0,1),label=("500kg"))

plt.ylabel('Force (N)')                               #Display Speed-Torque Curves
plt.xlabel('Slip angle (deg)')
plt.legend()
plt.show()

"""CAMBER EFFECTS"""
plt.plot(friction_data_slip_angle,Dataset_tire_friction_raw[21:36,2]*1000, label="Measured Data, 400kg, 0 CAMBER")
plt.plot(SA,MF_lateral_function(SA_rad,friction_data_weight[0],0,1),label="MF Tire model, 400kg, 0 CAMBER")
axes = plt.gca()
axes.set_ylim([-8000,8000])

plt.legend()
plt.show()
plt.plot(friction_data_slip_angle,Dataset_tire_friction_raw[21:36,14]*1000, label="Measured Data, 400kg, 3 CAMBER")
plt.plot(SA,MF_lateral_function(SA_rad,friction_data_weight[1],0.05,1),label="MF Tire model, 400kg, 3 CAMBER")
axes = plt.gca()
axes.set_ylim([-8000,8000])

plt.legend()
plt.show()