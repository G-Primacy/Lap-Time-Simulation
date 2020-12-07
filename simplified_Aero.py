# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:57:41 2020

@author: Shawn
"""
import numpy as np
import matplotlib.pyplot as plt

"Global Environment Parameters (SI)"
g=9.81;         "Gravity (m/s**2)"
rho_air=1.2754;        "Density of Air (kg/m**3)"

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

def downforce_solver(u):
    "Given an input velocity, computes and returns the downforce (N) for longitudinal velocity"
    downforce_local = 0.5*(rho_air)*(Area_Reference_SI)*C_L*u**2;    "Downforce (N) = 1/2*Fluid Density*Reference Area*Lift*Velocity^2"
    return downforce_local
def drag_solver(u):
    "Given an input velocity, computes and returns drag (N) for longitudinal velocity"
    drag_local=-0.5*(rho_air)*(Area_Reference_SI)*C_D*u**2;   "Downforce (N) = 1/2*Fluid Density*Reference Area*Lift*Velocity^2"
    return drag_local

def plot_simple_aero():
    "Plots simple 1D downforce and drag plot as function of longitudinal velocity"
    u=np.linspace(0,60,100)
    Downforce_local=downforce_solver(u)
    Drag_local=drag_solver(u)
    
    plt.plot(u,Downforce_local, label='Downforce (N)')
    plt.plot(u,Drag_local, label='Drag (N)')
    plt.title('0 DOF LONGITUDINAL TRAVEL DRAG & DOWNFORCE')
    plt.ylabel('Magnitude ()')
    plt.xlabel('Longitudinal Speed (m/s)')
    plt.legend()
    plt.show()
