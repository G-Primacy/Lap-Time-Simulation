# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:51:00 2020

@author: Shawn
"""

"""Feed the lap time simulation either:
    X, Y, Z data
    line and arc data
    """
#Plot Track

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
import math
import rdp
from scipy.interpolate import splprep, splev
#***************************************************************************************************************#
#***************************************************************************************************************#
"PANDAS IMPORT: SPREADSHEET DATA"
print("Solving: Pandas Imports")
#***************************************************************************************************************#
#***************************************************************************************************************#

trackdata_coords = np.asarray(pd.read_csv('G:/My Drive/The Race Car Project/Track Data/Pikes Peak International Hillclimb Cleaned.CSV'))

#plt.plot(trackdata_coords[1:633,1],trackdata_coords[1:633,2])
Data_size = np.size(trackdata_coords[1:633,1])



def length_solver(point_a,point_b):
    "Calculates the magnitude of length between to points"
    delta_i=point_a[0]-point_b[0]
    delta_j=point_a[1]-point_b[1]
    
    length=np.sqrt(delta_i**2+delta_j**2)
    
    return length

#Declaring arrays of X,Y,Z GPS coordinates
X_array_GPS = np.abs(trackdata_coords[:,1])     
Y_array_GPS = np.flip(np.abs(trackdata_coords[:,2]))
Z_array_GPS = np.abs(trackdata_coords[:,3])

def unit_vector_solver(point_a,point_b):
    "Calculates the unit vector between two points"
    "point_a: origin"
    "point_b: target"
    delta_i=point_b[0]-point_a[0]
    delta_j=point_b[1]-point_a[1]
    
    length=np.sqrt(delta_i**2+delta_j**2)
    
    unit_vector=[delta_i/length,delta_j/length]
    return unit_vector

def farthest_pair (X, Y):
    # test points
    pts = [X,Y]
    
    # two points which are fruthest apart will occur as vertices of the convex hull
    candidates = pts[spatial.ConvexHull(pts).vertices]
    
    # get distances between each pair of candidate points
    dist_mat = spatial.distance_matrix(candidates, candidates)
    
    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    
    print(candidates[i], candidates[j])
    # e.g. [  1.11251218e-03   5.49583204e-05] [ 0.99989971  0.99924638]
    return (pts)

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

def angle(dir):
    """
    Returns the angles between vectors.

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1*dir2).sum(axis=1)/(
        np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

#Scale lat/long to real life meters
Y_scale= 8800   #meters, rough google maps linear 
X_scale= 6900   #meters, rough google maps linear 

#Establish start and end points
start = [X_array_GPS[0],Y_array_GPS[0]]
end =   [X_array_GPS[Data_size],Y_array_GPS[Data_size]]

#Determine GPS longest length
longest_gps_length = length_solver(start,end)

#Calculate appropriate scaling factors for GPS coordinates
scale_factor_Y = Y_scale/longest_gps_length
scale_factor_X = X_scale/longest_gps_length

#GPS converted to metric
X_array_M = X_array_GPS*scale_factor_X
Y_array_M = Y_array_GPS*scale_factor_Y
Z_array_M = Z_array_GPS


#Find GPS coord limits
X_sorted = np.sort(X_array_M)
Y_sorted = np.sort(Y_array_M)
Z_sorted = np.sort(Z_array_M)

#Subtract Metric coordinates by smallest value to achieve zeros
X_array = X_array_M-X_sorted[0]
Y_array = Y_array_M-Y_sorted[0]
Z_array = Z_array_M-Z_sorted[0]

#Plot 2D track (metric)
plt.plot(X_array, Y_array, c='b', label='y1',linewidth=0.5)
plt.show()

#Plot 2D track (gps)
plt.plot(X_array_GPS, Y_array_GPS, c='b', label='y1',linewidth=0.5)
plt.show()

plt.savefig('G:/My Drive/The Race Car Project/Track Data/Pikes Peak International Hillclimb MATPLOTLIB2.jpg', dpi=1000, format='jpg')


#Axes Labels
plt.show()

i=0

"""SPLINE FIT THE DATA TO PRODUCE CONTINUOUS DRIVING LINE"""
#INTERP1D DOESNT WORK FOR TRACK DATA SINCE X VALUES CAN APPEAR TWICE AND CAN CROSS OVER
#Does not support closed curves

pts= np.asarray([X_array,Y_array])

tck, u = splprep(pts, s=0.0,k=3) #k= fit degree
u_new = np.linspace(u.min(), u.max(), 700) #number of points
x_new, y_new = splev(u_new, tck, der=0)


"""ALGORITHM: DETERMINE IF A SEQUENCE OF POINTS IS A STRAIGHT OR A TURN RDP METHOD"""

tolerance = 15
min_angle = np.pi*0.0005

points = np.asarray([x_new,y_new]).T
print(len(points))
x, y = points.T

# Use the Ramer-Douglas-Peucker algorithm to simplify the path
# http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
# Python implementation: https://github.com/sebleier/RDP/
simplified = np.array(rdp.rdp(points.tolist(), tolerance))

print(len(simplified))
sx, sy = simplified.T

# compute the direction vectors on the simplified curve
directions = np.diff(simplified, axis=0)
theta = angle(directions)
# Select the index of the points with the greatest theta
# Large theta is associated with greatest change in direction.
idx = np.where(theta>min_angle)[0]+1


"""PLOT DRIVING LINES"""

fig = plt.figure()
ax =fig.add_subplot(111)

ax.plot(X_array, Y_array, 'b-', label='original path')
ax.plot(sx, sy, 'g--', label='simplified path')
ax.plot(sx[idx], sy[idx], 'ro', markersize = 4, label='turning points')
ax.plot(x_new, y_new, 'y--', label='simpline path')
#ax.invert_yaxis()
plt.legend(loc='best')

axes = plt.gca()
axes.set_xlim([0,3500])
axes.set_ylim([0,9000])

plt.show()