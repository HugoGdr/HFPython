# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:09:01 2021

@author: Hugo
"""

import numpy as np
from numpy import *
import scipy
from scipy.special import erf

def list_diff(list_A, list_B):
    #make difference between the elements of two same size lists
    if len(list_A) != len(list_B):
        print("The two lists in list_diff have different sizes")
        exit
    list_C = []
    for i in range(len(list_A)):
        list_C.append(list_A[i]-list_B[i])
    return list_C

def xyz_reader(file_name):
    #Reads an xyz file and returns the number of atoms
    #atom types and coordinates
    
    file = open(file_name,'r')
    
    number_of_atoms = 0
    atom_type = []
    atom_coordinates = []
    
    for idx,line in enumerate(file):
        #Get number of atoms
        if idx == 0:
            try:
                number_of_atoms = int(line.split()[0])
            except:
                print("xyz file not in correct format")
                
        #Skip the comment/blank line
        if idx==1:
            continue
        
        #Get atom type and positions
        if idx != 0:
            split = line.split()
            atom = split[0]
            coordinates = [float(split[1]),
                           float(split[2]),
                           float(split[3])]
            
            atom_type.append(atom)
            atom_coordinates.append(coordinates)
            
    file.close()
            
    return number_of_atoms, atom_type, atom_coordinates

def gauss_product(gauss_A, gauss_B):
    #product of 2 gaussians is another gaussian
    
    a, Ra = gauss_A
    b, Rb = gauss_B
    p = a + b
    diff_Ra_Rb = list_diff(Ra,Rb)
    diff = np.linalg.norm(diff_Ra_Rb)**2
    N = (4*a*b/(pi**2))**0.75
    K = N*exp(-a*b/p*diff)
    Rp = []
    for i in range(len(Ra)):
        Rp.append((a*Ra[i]+b*Rb[i])/p)
    
    return p, diff, K, Rp

def overlap(A, B):
    #Overlap integral
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (pi/p)**1.5
    return prefactor*K

def kinetic(A,B):
    p, diff, K, Rp = gauss_product(A, B)
    prefactor = (pi/p)**1.5
    
    a, Ra = A
    b, Rb = B
    reduced_exponent = a*b/p
    return reduced_exponent*(3-2*reduced_exponent*diff)*prefactor*K

def Fo(t):
    #Fo function for calculating potential and e-e repulsion integrals
    if t == 0:
        return 1
    else:
        return (0.5*(pi/t)**0.5)*erf(t**0.5)
    
def potential(A,B,atom_idx):
    #Nuclear-electron integral
    p,diff,K,Rp = gauss_product(A,B)
    Rc = atom_coordinates[atom_idx] #Position of atom C
    Zc = charge_dict[atoms[atom_idx]] #Charge of atom C
    
    diff_Rp_Rc = list_diff(Rp,Rc)
    return (-2*pi*Zc/p)*K*Fo(p*np.linalg.norm(diff_Rp_Rc)**2)

def multi(A,B,C,D):
    #(ab|cd) integral
    p, diff_ab, K_ab, Rp = gauss_product(A,B)
    q, diff_cd, K_cd, Rq = gauss_product(C,D)
    multi_prefactor = 2*pi**2.5*(p*q*(p+q)**0.5)**-1
    diff_Rp_Rq = list_diff(Rp,Rq)
    return multi_prefactor*K_ab*K_cd*Fo(p*q/(p+q)*np.linalg.norm(diff_Rp_Rq)**2)

def SD_successive_density_matrix_elements(Ptilde,P):
    x = 0
    for i in range(B):
        for j in range(B):
            x += B**-2*(Ptilde[i,j]-P[i,j])**2
            
    return x**0.5

def get_nuclear_repulsion():
    Nuc_repuls = 0
    for idx_a, A in enumerate(atoms):
        for idx_b, B in enumerate(atoms):
            if idx_a == idx_b:
                continue
            charge_A = charge_dict[A]
            charge_B = charge_dict[B]
            product = charge_A * charge_B
            Ra = atom_coordinates[idx_a]
            Rb = atom_coordinates[idx_b]
            diff_Ra_Rb = list_diff(Ra,Rb)
            R = np.linalg.norm(diff_Ra_Rb)
            Nuc_repuls += product/R
    return Nuc_repuls*0.5
        
# Basis set variables

#STO-nG (number of gaussians used to form a contracted gaussian orbital)
STOnG = 3

#Dictionary of zeta values
zeta_dict = {'H':[1.24], 'He':[2.0925], 'Li':[2.69,0.80], 'Be':[3.68,1.15],
             'B':[4.68,1.50], 'C':[5.67,1.72]}

#Dictionary containing the max quantum number of each atom for a minimal
#basis STO-nG calculation

max_quantum_number = {'H':1, 'He':1, 'Li':2, 'Be':2, 'C':2}

#Gaussian contraction coefficients
#Row represents 1s, 2s,...
D = np.array([[0.444635, 0.535328, 0.154329],
              [0.700115, 0.399513,-0.0999672]])

#Gaussian orbital exponents
alpha = np.array([[0.109818, 0.405771, 2.22766],
                  [0.0751386, 0.231031, 0.994203]])

    
#Number of electrons
N = 2

#Dictionary of charges
charge_dict = {'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10}


file_name = 'HeH.xyz'

N_atoms, atoms, atom_coordinates = xyz_reader(file_name)

#Basis set size
B = 0
for atom in atoms:
    B += max_quantum_number[atom]
    
# Initialise matrices
S = np.zeros((B,B))
T = np.zeros((B,B))
V = np.zeros((B,B))

multi_electron_tensor = np.zeros((B,B,B,B))

#Iterate through atoms
for idx_a, val_a in enumerate(atoms):
    
    #For each atom , get the charge and the centre
    Za = charge_dict[val_a]
    Ra = atom_coordinates[idx_a]
    
    #Iterate through quantum numbers (1s, 2s...)
    for m in range(max_quantum_number[val_a]):
        #For each Q number, get contraction coeff, zeta, scale expo
        d_vec_m = D[m]
        zeta = zeta_dict[val_a][m]
        alpha_vec_m = alpha[m]*zeta**2
        
        #Iterate over the contraction coeff
        for p in range(STOnG):
            
            #Iterate through atoms once again
            for idx_b, val_b in enumerate(atoms):
                Zb = charge_dict[val_b]
                Rb = atom_coordinates[idx_b]
                
                for n in range(max_quantum_number[val_b]):
                    d_vec_n = D[n]
                    zeta = zeta_dict[val_b][n]
                    alpha_vec_n = alpha[n]*zeta**2
                      
                    for q in range(STOnG):
                        a = (idx_a+1)*(m+1)-1
                        b = (idx_b+1)*(n+1)-1
                          
                        #Generate overlap, kinetic and potential matrices
                        S[a,b] += d_vec_m[p]*d_vec_n[q]*overlap((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb))
                        T[a,b] += d_vec_m[p]*d_vec_n[q]*kinetic((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb))
                          
                        for i in range(N_atoms):
                            V[a,b] += d_vec_m[p]*d_vec_n[q]*potential((alpha_vec_m[p],Ra),(alpha_vec_n[q],Rb),i)
                            
                            #2 more iterations to get the multi-electron tensor
                            for idx_c, val_c in enumerate(atoms):
                                Zc = charge_dict[val_c]
                                Rc = atom_coordinates[idx_c]
                                for k in range(max_quantum_number[val_c]):
                                    d_vec_k = D[k]
                                    zeta = zeta_dict[val_c][k]
                                    alpha_vec_k = alpha[k]*zeta**2
                                    for r in range(STOnG):
                                        for idx_d, val_d in enumerate(atoms):
                                            Zd = charge_dict[val_d]
                                            Rd = atom_coordinates[idx_d]
                                            for l in range(max_quantum_number[val_d]):
                                                d_vec_l = D[l]
                                                zeta = zeta_dict[val_d][l]
                                                alpha_vec_l = alpha[l]*zeta**2
                                                for s in range(STOnG):
                                                    c = (idx_c+1)*(k+1)-1
                                                    d = (idx_d+1)*(l+1)-1
                                                    multi_electron_tensor[a,b,c,d] += d_vec_m[p]*d_vec_n[q]*d_vec_k[r]*d_vec_l[s]*\
                                                    multi((alpha_vec_m[p],Ra),
                                                          (alpha_vec_n[q],Rb),
                                                          (alpha_vec_k[r],Rc),
                                                          (alpha_vec_l[s],Rd))
                                                    
Hcore = T + V

#Symmetric orthogonalisation of basis
evalS, U = np.linalg.eig(S)
diagS = dot(U.T,dot(S,U))
diagS_minushalf = diag(diagonal(diagS)**-0.5)
X = dot(U,dot(diagS_minushalf,U.T))

#Initial guess at P
P = np.zeros((B,B))
P_previous = np.zeros((B,B))
P_list = []

#Iterative process
conv = []
threshold = 100
while threshold > 10**-4:
    #Calculate Fock matrix with guess
    G = np.zeros((B,B))
    for i in range(B):
        for j in range(B):
            for x in range(B):
                for y in range(B):
                    G[i,j] += P[x,y]*(multi_electron_tensor[i,j,x,y]-0.5*multi_electron_tensor[i,x,y,j])
    Fock = Hcore + G
    
    #Calculate Fock matrix in orthogonalised base
    Fockprime = dot(X.T,dot(Fock,X))
    evalFockprime, Cprime = np.linalg.eig(Fockprime)
    
    #Correct ordering of eigenvalues and eigenvectors
    idx = evalFockprime.argsort()
    evalFockprime = evalFockprime[idx]
    Cprime = Cprime[:,idx]
    
    C = dot(X,Cprime)
    
    #Form New P
    for i in range(B):
        for j in range(B):
            for a in range(int(N/2)):
                P[i,j] = 2*C[i,a]*C[j,a]
                
    P_list.append(P)
    
    threshold = SD_successive_density_matrix_elements(P_previous,P)
    P_previous = P.copy()
    conv.append(threshold)
    
Nuc_repuls=get_nuclear_repulsion()
    
print('\n')
print('STO3G Restricted Closed Shell HF algorithm took {} iterations to converge'.format(len(P_list)))
print('\n')
print('The orbital energies are {} and {} Hartrees'.format(evalFockprime[0],evalFockprime[1]))
print('\n')
print('The orbital matrix is: {}'.format(C))
print('\n')
print('The density/bond order matrix is: {}'.format(P))
print('\n')
print('Nuclear repulsion is : {}'.format(Nuc_repuls))