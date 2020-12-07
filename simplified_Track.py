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
from scipy.ndimage.interpolation import shift

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


#Plot 3D track(GPS)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax = plt.axes(projection='3d')
#Plot 3d track
ax.scatter(xs = X_array, 
               ys = Y_array, 
               zs = Z_array,
               s=5,
               zdir='z',
               cmap='r')

#Axes Labels
plt.title('Track Map')
ax.set_ylabel('Y (m)')                         
ax.set_xlabel('X (m)')
ax.set_zlabel('Z (m)')
#Save File
fig.savefig('G:/My Drive/The Race Car Project/Track Data/Pikes Peak International Hillclimb MATPLOTLIB.jpg', dpi=1000, format='jpg')

i=0

"""SPLINE FIT THE DATA TO PRODUCE CONTINUOUS DRIVING LINE"""
#INTERP1D DOESNT WORK FOR TRACK DATA SINCE X VALUES CAN APPEAR TWICE AND CAN CROSS OVER
#Does not support closed curves

pts= np.asarray([X_array,Y_array])

tck, u = splprep(pts, s=0,k=3) #k= fit degree, s = smoothing
u_new = np.linspace(u.min(), u.max(), 1500) #number of points
x_spline, y_spline = splev(u_new, tck, der=0)


"""ALGORITHM: DETERMINE IF A SEQUENCE OF POINTS IS A STRAIGHT OR A TURN RDP METHOD"""

tolerance = 20
min_angle = np.pi*0.001

points = np.asarray([x_spline,y_spline]).T
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


"""PLOT DRIVING LINES + TURNING POINTS"""

fig = plt.figure()
ax =fig.add_subplot(111)

ax.plot(X_array, Y_array, 'b-', label='original path')
#ax.plot(sx, sy, 'g--', label='simplified path')
#ax.plot(sx[idx], sy[idx], 'ro', markersize = 4, label='turning points')
#ax.plot(x_spline, y_spline, 'y--', label='spline path')
#ax.invert_yaxis()
plt.legend(loc='best')

plt.title('Track Map')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
axes = plt.gca()
axes.set_xlim([0,3500])
axes.set_ylim([0,9500])

plt.show()


fig = plt.figure()
ax =fig.add_subplot(111)

#ax.plot(X_array, Y_array, 'b-', label='original path')
ax.plot(sx, sy, 'g--', label='simplified path')
ax.plot(sx[idx], sy[idx], 'ro', markersize = 4, label='turning points')
#ax.plot(x_spline, y_spline, 'y--', label='spline path')
#ax.invert_yaxis()
plt.legend(loc='best')

plt.title('Track Map')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
axes = plt.gca()
axes.set_xlim([0,3500])
axes.set_ylim([0,9500])

plt.show()


fig = plt.figure()
ax =fig.add_subplot(111)

#ax.plot(X_array, Y_array, 'b-', label='original path')
#ax.plot(sx, sy, 'g--', label='simplified path')
ax.plot(sx[idx], sy[idx], 'ro', markersize = 4, label='turning points')
ax.plot(x_spline, y_spline, 'y--', label='Processed Path')
#ax.invert_yaxis()
plt.legend(loc='best')

plt.title('Track Map')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
axes = plt.gca()
axes.set_xlim([0,3500])
axes.set_ylim([0,9500])

plt.show()

"""CORELATE TURN ANGLES AND TURN RADII"""

#Minimum turn radii/
r_min = 10
#radii growth with turn departure angle
#180 deg = 10m
#90 deg = 20m
#45 deg = 40m
#30 deg = 100m

fig = plt.figure()
ax =fig.add_subplot(111)

radii = np.asarray([10, 20, 40, 100,50,    50, 20, 50, 120,    200,    75])
angle = np.asarray([180,90, 45, 20, 125,    70, 150,90, 10,     5,      90])
angle_rad = angle*(1/180*np.pi)
angles = np.linspace(0,np.pi,50)

def turn_rad_poly(x):
    #Returns approproiate turn radius given an angle between input vector and departure vector of a simplified turn
    A = 20
    B = 10
    C = 1.5
    y =A/(x**C)+B
    
    return y

plt.scatter(angle_rad, radii) #points
plt.plot(angles, turn_rad_poly(angles)) #function

plt.ylabel('turn radois (m)')
plt.xlabel('angle (rad)')

axes = plt.gca()
axes.set_xlim([0,5])
axes.set_ylim([0,150])
plt.show()



"""CALCULATING TURN ANGLES"""
i=0
track_siimplified = np.asarray([sx,sy])     #Array of simplified X,y coords
track_siimplified_T = track_siimplified.T   #Transposed simplified X,Y array
track_lengths = np.zeros(np.size(sx))       #Array of lengths between X,Y coords
s=(np.size(sx),2)                           #Array shape for array of unit vectors
unit_vectors = np.zeros(s)                  #array of unit vectors
turn_angles = np.zeros(np.size(sx))
#Calculate lengths between simplified points
while i < (np.size(sx)-1):
    track_lengths[i] = length_solver(track_siimplified_T[i], track_siimplified_T[i+1])
    i+=1
#Calculate unit vectors of each length
i=0
while i < (np.size(sx)-1):
    unit_vectors[i] = unit_vector_solver(track_siimplified_T[i], track_siimplified_T[i+1])
    i+=1
#Calculate the angle between each unit vector
i=0
while i < (np.size(sx)-2):
    turn_angles[i]=UV_angle_solver(unit_vectors[i], unit_vectors[i+1])
    i+=1
    
#plot test section
test_size = 24
fig = plt.figure()
ax =fig.add_subplot(111)

plt.plot(sx[0:test_size], sy[0:test_size], 'r--',linewidth=0.5, label='simplified path')
plt.scatter(sx[idx[0:test_size]], sy[idx[0:test_size]],marker='o',c=turn_angles[0:test_size], s = 15, label='turning points')
plt.legend(loc='best')

plt.title('Track Map')
plt.xlabel('x (m)')
plt.ylabel('y (m)')


axes = plt.gca()
axes.set_xlim([0,1555])
axes.set_ylim([0,1555])

plt.show()

"""CALCULATING TURN RADII FOR TURN ANGLES"""
turn_radii = np.zeros(np.size(sx))
turn_type = list()
i=0
while i < (np.size(sx)):
    #If the angle is small enough, asume negligable angle
    negligable_angle = np.pi/6
    
    if turn_angles[i] < negligable_angle:
        turn_radii[i] = 0
    else:
        turn_radii[i]=turn_rad_poly(turn_angles[i])
    print(turn_radii[i])
    #If the angle is small enough, asume negligable angle
    i+=1
    
"""CALCULATING TRUNCATED STRAIGHT LENGTHS - TURN RADII"""
track_lengths_truncated = track_lengths - turn_radii # Turn radii subtracted from end of entry straight
#Negative resultant lengths imply that two turns are too close to one another (slalom)
#OR connected turns, or extended turns
#We will fudge it by absoluting negative values since they are so negligable (2-10m overshoot with 1-3 occurancecs in 150 segments)
turn_arc_length = turn_radii * turn_angles
i=0
#CONVERT NEGATIVE LENGTHS TO ZERO
while i < np.size(track_lengths_truncated):
    if track_lengths_truncated[i] < 0:
        track_lengths_truncated[i]=0
    i+=1

"""PRODUCE PROCESSED TRACK DATA"""
#track length*4 2D array for | sement length | radius | turn angle | segment type
Track_segments_processed_size = [np.size(track_lengths_truncated) + np.size(turn_arc_length),4]
Track_segments_processed = np.zeros(Track_segments_processed_size)
i=0 #track_lengths_truncated iterator
j=0 #Track_segments_processed iterator

#EVEN NUMBERS: STRAIGHTS
#ODD NUMBERS: TURNS
while j < Track_segments_processed_size[0]:
    Track_segments_processed[j,0] = track_lengths_truncated[i]  #Segment straight length
    Track_segments_processed[j,3] = 1                           #Segment state
    
    Track_segments_processed[j+1,0] = turn_arc_length[i]        #Segment turn arc length
    Track_segments_processed[j+1,1] = turn_radii[i]        #Segment turn arc length
    
    Track_segments_processed[j+1,2] = turn_angles[i]        #Segment turn arc length
    Track_segments_processed[j+1,3] = 2                     #Segment state
    
    j+=2
    i+=1
    
np.savetxt("TrackData.csv",Track_segments_processed, delimiter=",")