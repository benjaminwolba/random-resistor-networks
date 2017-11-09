#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 2D DC Random Resistor Networks, Benjamin Wolba

    Provides functions for solving 2D DC random resistor networks (RRN).
"""

# Used modules

from __future__ import division

import numpy as np
import numpy.random as rd
from scipy.sparse.linalg import spsolve


# Creation of a conductor matrix

def make_conductors(p, n, sig_met, sig_ins):
    """ Creates one array for the metallic conductors and another one for 
        the insulating ones, according to the metallic fraction p. These are
        combined, shuffeld and returned in the final conductor matrix.
                
    Parameter:
        p:          metallic fraction
        n:          number of conductors per row
        sig_met:    conductivity of metallic conductors
        sig_ins:    conductivity of insulating conductors
    
    Return:
        cond_matr:  conductor matrix
    """
    
    num = (2*n)*(n)
    
    met_matr = sig_met*np.ones(round(p*num))        # metallic conductors
    ins_matr = sig_ins*np.ones(round((1-p)*num))    # insulating conductors
    
    cond_matr = np.hstack((met_matr, ins_matr))     # combine
    cond_matr = rd.choice(cond_matr, size=(2*n,n))  # shuffle + reshape      
    
    return(cond_matr)


# Exact solution through a conductivity matrix

def make_cond_matr(cond_matr, n):
    """ Sets up the conductivity matrix G for a given conductor matrix.
                
    Parameter:
        cond_matr:    conductor matrix
        n:            number of conductors per row
    
    Return:
        G_matr:    conductivity matrix
    """
  
    for i in range(n):
        cond_matr[2*i,int(n)-1] = 0.0
    
    n_sq = n**2
    G_matr = np.zeros((n_sq,n_sq)) 
    
    # Sort conductors into the G-matrix
    
    for i in range(n_sq):
        k, l = i%n, (i-i%n)/n 
        
        G_matr[i, (i+1)%n+(i-i%n)] = cond_matr[(2*k+1),l]
        G_matr[i, (i-1)%n+(i-i%n)] = cond_matr[(2*k-1)%(2*n),l]
        G_matr[i, (i+n)%n_sq] = cond_matr[(2*k),l]
        G_matr[i, (i-n)%n_sq] = cond_matr[(2*k),l-1] 
        
        G_matr[i, i] = -np.sum(G_matr[i, :])
    
    return(G_matr)


def solve_LGS(G_matr, n, V_c):
    """
    Main program to solve the Kirchhoff equations.
    
    Parameter:
        G_matr:    conductivity matrix
        n:         number of conductors per row
        V_c:       applied potential
    
    Return:
        rho:    total network resistance    
    """
    
    n_sq = n**2
    U_arr = np.hstack((np.zeros(n_sq-n), V_c*np.ones(n)))
    c_arr = np.zeros(n_sq-2*n)

    # Solution of the reduced problem
    
    c_arr = np.dot(G_matr[n:n_sq-n,:], U_arr)    
        
    G_red = G_matr[n:n_sq-n,n:n_sq-n]
    
    U_arr[n:n_sq-n] = spsolve(G_red, -c_arr)
    
    
    # Calculation of currents and the total network resistance
    
    I_arr = np.dot(G_matr, U_arr)
    I_ges = np.sum(I_arr[:n])
    rho = V_c/I_ges
    
    return(rho) 
    

# Numerical solution through the relaxation method     

def relaxation(V, cond_matr, n):
    """ Using Ohms Law and current conservation, the basis of Kirchoffs Laws,
        this function conducts one relaxations step for a given potential
        array V and thus updates it to the new potential values.
                
    Parameter:
        V:            initial potential matrix
        cond_matr:    matrix of conductors
        n:            number of conductors per row
    """
    
    for i in range(0,n):
        for j in range(1,n-1):
            
            # Sum over the potentials of neighboring nodes, weighted by the
            # conductivity of the connecting conductors:        
            numerator = cond_matr[(2*i+1),j]*V[(i+1)%n,j] + \
                        cond_matr[(2*i-1)%(2*n),j]*V[i-1,j] + \
                        cond_matr[(2*i),j]*V[i,j+1] + \
                        cond_matr[(2*i),j-1]*V[i,j-1]
                        
            # Sum over the conductivities included above:
            denominator = cond_matr[(2*i+1),j] + cond_matr[(2*i-1)%(2*n),j] +\
                          cond_matr[(2*i),j] + cond_matr[(2*i),j-1]
                          
            V[i,j] = numerator/denominator


def solve_RRN(cond_matr, eps, n, V_0 = 0.0, V_c = 10.0):
    """ Solves the Kirchoff equations by conducting relaxation steps until
        the potential difference between two consecutive V-matrices is less
        than eps for every matrix element. Then the currents for all
        columns are calculated and returned together with the counter (i.e.
        number) of relaxation steps.
    
    Parameter:
        cond_matr:    conductor matrix
        eps:          required accuracy
        n:            number of conductors per row
        V_0:          reference potential 
        V_c:          applied potential
    
    Return:
        j_av_1:       currents for every column
        counter_1:    number of relaxation steps    
    """
    
    # Set up the potential matrix
    
    V_matr_1 = np.ones((n,n))*np.linspace(V_0, V_c, n)
    
    dif_1_1 = 1.0                                   # initial difference
    counter_1 = 0                                   # initial counter

    # Relaxation process
    
    while dif_1_1 > eps:
        V_new_1 = np.copy(V_matr_1)
        relaxation(V_new_1, cond_matr, n)              # relaxation step
        dif_1_1 =  np.amax(np.abs(V_new_1 - V_matr_1)) # difference to former
                                                       # voltage matrix
        V_matr_1 = V_new_1
        counter_1 += 1                                 # increase counter


    # Calculation of the current density j

    j_arr_1 = np.zeros(n-1)                         # current densities array
    
    for j in range(n-1):                            # calculate the current
        for i in range(n):                          # densitiy for every column
            j_arr_1[j] += cond_matr[(2*i),j]*( \
            V_matr_1[i,j+1]-V_matr_1[i,j])
               
    j_av_1 = np.sum(j_arr_1)/(n-1)                  # average current density
    
    return(j_av_1, counter_1)
