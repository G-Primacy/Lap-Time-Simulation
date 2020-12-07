# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:01:55 2020

@author: Shawn
"""

def centripetal_foce (mass, radius, velocity):
    #mass: kg
    #radius: m
    #velocity: m/s
    force = mass*velocity/radius**2
    return force