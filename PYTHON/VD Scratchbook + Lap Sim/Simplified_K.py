# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 23:23:32 2020

@author: Shawn

A Simplified suspension position and kinematics solver for motion ratios, camber curves, steering forces, roll steer, roll gradients.

Control arm motion solver (camber gain, roll centers, inst. center, jacking)
Direct suspension solver (motion ratios, spring force curves damper curves)
Pushrod/Pull rod solver (motion ratios, spring force curves damper curves)

"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import *



"""Print Derivation Media"""
from IPython.display import display, Image

display(Image(filename='Import Media/Suspension Kinematics.jpg'))
display(Image(filename='Import Media/Suspension Kinematics2.jpg'))


def length_solver(point_a,point_b):
    "Calculates the magnitude of length between to points"
    delta_i=point_a[0]-point_b[0]
    delta_j=point_a[1]-point_b[1]
    
    length=np.sqrt(delta_i**2+delta_j**2)
    
    return length

def unit_vector_solver(point_a,point_b):
    "Calculates the unit vector between two points"
    "point_a: origin"
    "point_b: target"
    delta_i=point_b[0]-point_a[0]
    delta_j=point_b[1]-point_a[1]
    
    length=np.sqrt(delta_i**2+delta_j**2)
    
    unit_vector=[delta_i/length,delta_j/length]
    
    return unit_vector
def angle_to_unit_vector(angle):
    
    unit_vector=[np.cos(angle),np.sin(angle)]
    return(unit_vector)

def unit_vector_to_angle(unit_vector):
    angle=np.arccos(unit_vector[0])
    return(angle)

def cosine_angle_solver(length_a,length_b,length_c):
    "Calculates an angle with 3 known lengths, cosine rule"
    theta=np.arccos(
        (length_b**2
         +length_c**2-length_a**2)/(2*length_b*length_c)
        )
    return(theta)

def known_lengths_angle_solver (length_A,length_B,length_C):
    "computes the angles opposite of lengths"
    "theta_a is opposite of length a" """
        _____A_____
      /c      ___b/
    B/   ____/C
    /a__/                    """
    theta_a=cosine_angle_solver(length_A,length_B,length_C)
    theta_b=cosine_angle_solver(length_B,length_A,length_C)
    theta_c=cosine_angle_solver(length_C,length_B,length_A)
    return(theta_a,theta_b,theta_c)

def UV_angle_solver(UV_a, UV_b):
    "Calculates the angle between two vectors via dot product derivation"
    "DOUBLE CHECK THE ORIGINS"
    #Dot Product: |a||b|cos(theta)=dot(a,b)
    angle =np.arccos(
        (np.dot(UV_a,UV_b)/
        (np.sqrt(UV_a[0]**2+UV_a[1]**2)*np.sqrt(UV_b[0]**2+UV_b[1]**2))
        )
        )
  
    
    return(angle)

def unit_vector_transform(UV,angle):
    "Calculates transformed vector from base vector and transform angle"
    UV_rot=[UV[0]*np.cos(-angle) + UV[1]*np.sin(-angle),
             -UV[0]*np.sin(-angle) + UV[1]*np.cos(-angle)]
    return(UV_rot)

def camber_solver(point_loop_local):
    "Computes the camber based on upright angular position"
    point_LCA_outer_local=point_loop_local[1]
    point_UCA_outer_local=point_loop_local[2]
    UV_upright_local=unit_vector_solver(point_LCA_outer_local, point_UCA_outer_local)
    
    angle_ground=[0,1]      #Purely vertical unit vector, angle = pi/2rad
    angle_camber=-UV_angle_solver(angle_ground,UV_upright_local)   #Angle between the upright and pure vertical vector
    "angle is always produced positive magnitude, must correct for direction"
    "positive: UVB is CCW of UVA"
    "negative: UVB is CW of UVA"
    if UV_upright_local[0] > 0:
        angle_camber=np.abs(angle_camber)
    else:
        angle_camber=-np.abs(angle_camber)
    
    return (angle_camber)

def point_transform(origin, unit_vector, length):
    point=[origin[0]+unit_vector[0]*length,origin[1]+unit_vector[1]*length]
    return (point)

def relative_POI_solver (point_interest, point_a, point_b):
    "Calculates the length, unit vector, of a point of intereset relative to point A, and AB vector"
    "Used for tire contact patch and rod joint"
    length=length_solver(point_interest,point_a)                #Calculates the length between point A, and POI
    unit_vector=unit_vector_solver(point_a,point_interest)      #Calculates the unit vector between point A, and POI
    angle = UV_angle_solver(unit_vector,unit_vector_solver(point_a,point_b))  #Calculates the fixed angle between A-B and A-POI
    return (length,unit_vector,angle)

def new_LCA_position (delta_Z, point_LCA_inner, length_LCA, unit_vector):
    "Calculates the unit vector of the new driven LCA position moved by a distance delta_Z vertical"
    control_point=[point_LCA_inner[0]+unit_vector[0]*length_LCA, point_LCA_inner[1]+unit_vector[1]*length_LCA]    #Coordinate of old joint position
    
    j=control_point[1]+delta_Z                                          #Coordinate of new joint position in vertical
    i=np.sqrt(length_LCA**2-delta_Z**2)+point_LCA_inner[0]                  #Coordinate of new joint position in horizontal  
    unit_vector_new=unit_vector_solver(point_LCA_inner,[i,j])           #unit vector of new LCA vector
    point_new=[i,j]                                                     #New 2D point coordinate of joint position
        
    return(unit_vector_new,point_new)

def plot_vector_loop(point_loop_local,point_rod_joint,point_tire_contact):
    "Plots vector loop and prismatic joints"
    
    plt.scatter(point_loop_local[:,0],point_loop_local[:,1],color='r')        #Plots control arm suspension loop
    plt.scatter(point_rod_joint[0],point_rod_joint[1],color='b')                #Plots suspension pickup point
    plt.scatter(point_tire_contact[0],point_tire_contact[1],color='b')                    #Plots contact patch point
    plt.plot(point_loop_local[:,0],point_loop_local[:,1],color='b')                     #Plots vector lines for suspension loop
    
    plt.grid(b=True, which='major', axis='both')
    plt.title('Quarter Suspension')
    plt.ylabel('z axis')
    plt.xlabel('y axis')
    axes = plt.gca()
    axes.set_xlim([000,700])
    axes.set_ylim([0,500])
    
    
"Solving Initial lengths, unit vector relative angles"
"lengths: Upper control arm (LCA), Lower control arm (UCA), inboard mounting, upright"
"Unit Vectors:  Upper control arm (LCA), Lower control arm (UCA), inboard mounting, upright"
"Angles (RAD): "
#***********************************************************************************************#
"""Control Arm Solver"""
print("Solving Control Arm Initial Conditions")
#***********************************************************************************************#

"Initial control arm positions"
point_LCA_inner=[206,174]
point_LCA_outer=[592,145]
point_UCA_outer=[591,371]
point_UCA_inner=[287,319]
point_rod_joint=[565,165]
point_tire_contact=[676,0]
point_loop=np.asarray([point_LCA_inner,point_LCA_outer,point_UCA_outer,point_UCA_inner,point_LCA_inner])

"Initial suspension positions"
point_rocker_pivot=[136,409]
point_rocker_input=[154,438]
point_rocker_output=[116,437]
suspension_loop=np.asarray([point_rod_joint,point_rocker_input,point_rocker_pivot,point_rocker_output])

length_LCA, UV_LCA=             length_solver(point_LCA_inner, point_LCA_outer),unit_vector_solver(point_LCA_inner, point_LCA_outer)
length_UCA, UV_UCA=             length_solver(point_UCA_inner, point_UCA_outer),unit_vector_solver(point_UCA_outer, point_UCA_inner)
length_mounting,UV_mounting=    length_solver(point_UCA_inner, point_LCA_inner),unit_vector_solver(point_UCA_inner, point_LCA_inner)
length_upright, UV_upright =    length_solver(point_UCA_outer, point_LCA_outer),unit_vector_solver(point_LCA_outer, point_UCA_outer)
length_rod, UV_rod =            length_solver(point_rod_joint, point_LCA_inner),unit_vector_solver(point_LCA_inner, point_rod_joint)
length_tire,UV_tire =           length_solver(point_tire_contact, point_LCA_outer),unit_vector_solver(point_LCA_outer, point_tire_contact)
angle_rod_LCA=UV_angle_solver(UV_LCA, UV_rod)
angle_tire_upright=UV_angle_solver(UV_upright, UV_tire)

"computing initial conditions: Suspension"
length_pushrod = length_solver(point_rod_joint,point_rocker_input)
length_input=   length_solver(point_rocker_input,point_rocker_pivot)
length_output=  length_solver(point_rocker_pivot,point_rocker_input)
angle_rocker = UV_angle_solver(unit_vector_solver(point_rocker_input,point_rocker_pivot), unit_vector_solver(point_rocker_pivot,point_rocker_output))



#**********************************************************************************************************************************#
"Calculating new position"
#**********************************************************************************************************************************#

delta_range=30; #Wheel driving distance
delta_step_quantity = 5

def control_arm_position_solver(delta_z,point_loop_local,point_rod_joint_local,point_tire_contact_local):
    "Solves the vector loop for a given change of tire contact movement from initial condition"
    point_LCA_inner_local=point_loop_local[0]       #Reassining points of point loop
    point_LCA_outer_local=point_loop_local[1]
    point_UCA_outer_local=point_loop_local[2]
    point_UCA_inner_local=point_loop_local[3]
    
    "Calculate driven LCA position"
    UV_LCA_new,point_LCA_outer_local=new_LCA_position(delta_z,point_LCA_inner, length_LCA, UV_LCA)    #Calculate new LCA position
    
    "Compute intermediate length between LCA outer and UCA inner. Divides Suspension intwo two triangles"
    length_intermediate=length_solver(point_LCA_outer_local,point_UCA_inner_local)

    "Compute all unknown angles"
    beta,angle_d,angle_a=known_lengths_angle_solver(length_intermediate,length_mounting,length_LCA)
    angle_e,angle_b,angle_c=known_lengths_angle_solver(length_intermediate,length_upright,length_UCA)
    "Internally, all angles are known in this control arm. We can now conpute where all of the unit vectors point."
    "Solving new UCA vector: UCA UV transformed via angle_a+ angle b"
    UV_UCA_REVERSED = unit_vector_transform(UV_mounting, (angle_a+angle_b))
    point_UCA_outer_local=point_transform(point_UCA_inner_local,UV_UCA_REVERSED,length_UCA)
    
    "Solving New upright, UCA vectors"
    UV_UCA=         unit_vector_solver(point_UCA_outer_local, point_UCA_inner_local)
    UV_upright =    unit_vector_solver(point_LCA_outer_local, point_UCA_outer_local)
    "Solving new rod joint position"
    UV_rod=unit_vector_transform(UV_LCA_new, angle_rod_LCA)
    point_rod_joint=point_transform(point_LCA_inner_local,UV_rod,length_rod)
    "Solving new tire contact position"
    UV_tire=unit_vector_transform(UV_upright, -angle_tire_upright)
    point_tire_contact=[point_LCA_outer_local[0]+UV_tire[0]*length_tire, point_LCA_outer_local[1]+UV_tire[1]*length_tire]
    
    "determining new point_loop positions"
    point_LCA_outer_local=point_transform(point_LCA_inner_local,UV_LCA_new,length_LCA)
    point_UCA_outer_local=point_transform(point_LCA_outer_local,UV_upright,length_upright)
    
    "Reassigning vector loop"
    point_loop_local=np.asarray([point_LCA_inner,point_LCA_outer_local,point_UCA_outer_local,point_UCA_inner,point_LCA_inner])  

    return (point_loop_local, point_rod_joint,point_tire_contact)

#**********************************************************************************************************************************#
"PUSH/PULL ROD SOLVER"
#**********************************************************************************************************************************#

def suspension_solver(point_rod,loop):
    point_rod
    point_rocker_input_local=     loop[1]
    point_rocker_pivot_local=     loop[2]
    point_rocker_output_local=    loop[3]


    "Solving driven position: pushrod"
    intermediate_length = length_solver(point_rocker_pivot_local,point_rod)
    UV_intermediate=unit_vector_solver(point_rod, point_rocker_pivot_local)
    angle_intermediate = UV_angle_solver([1,0],UV_intermediate)
    angle_theta, angle_phi, angle_psi=known_lengths_angle_solver(length_pushrod,length_input,intermediate_length)
    angle_pushrod=angle_intermediate-angle_phi
    UV_pushrod=[np.cos(angle_pushrod),np.sin(angle_pushrod)]
    
    "Solving driven position: rocker, input arm"
    point_rocker_input_new=point_transform(point_rod,UV_pushrod,length_pushrod)
    UV_rocker_input = unit_vector_solver(point_rocker_input_new,point_rocker_pivot_local)
    
    "solving driven position; output arm"
    UV_rocker_output = unit_vector_transform(UV_rocker_input,(2*np.pi-angle_rocker))
    point_rocker_output_new = point_transform(point_rocker_pivot_local,UV_rocker_output, length_output)
    print("Rocker Output: ", angle_theta)
    
    "Plotting initial condition"
    point_loop_pushrod=np.asarray([point_rod,point_rocker_input_new,point_rocker_pivot_local,point_rocker_output_new])
    plot_vector_loop(point_loop_pushrod,point_rod,point_tire_contact)
    
    return(point_rocker_output_new)


delta_step=delta_range/delta_step_quantity
delta_z=-delta_range
z_array=np.zeros(delta_step_quantity*2)
cambers=np.zeros(delta_step_quantity*2)
spring_displacement=np.zeros(delta_step_quantity*2)
i=0
while delta_z < delta_range:
    point_loop_new, point_rod_joint,point_tire_contact=control_arm_position_solver(delta_z,point_loop, point_rod_joint,point_tire_contact)
    plot_vector_loop(point_loop_new,point_rod_joint,point_tire_contact)
    ptc=suspension_solver(point_rod_joint,suspension_loop)
    
    """print("ROD POS: ", point_rod_joint)
    print("DELTA_Z: ", delta_z)"""
    
    cambers[i]=camber_solver(point_loop_new)
    z_array[i]=point_tire_contact[1]
    spring_displacement[i]=ptc[0]-point_rocker_output[0]
    delta_z = delta_z + delta_step
    
    i+=1
    

    plt.show()
cambers_deg=cambers*180/np.pi
plt.title("Camber as function of wheel travel.")
plt.plot(z_array,cambers_deg)
plt.show()

plt.title("Rocker delta Y as function of wheel travel.")
plt.plot(z_array,spring_displacement)
plt.show()
