#!/usr/bin/env python
# coding: utf-8

# # Background

# Computational physics is a field of study that uses numerical methods and computer simulations to solve complex physical problems that cannot be solved analytically. This approach allows researchers to study physical systems that are too large, too small, too complex, or too dangerous to be studied using traditional experimental methods.
# 
# Python is a powerful programming language used in computational physics to solve complex physical problems numerically. The time-independent Schrödinger equation for a particle in a one-dimensional potential well is a classic problem in quantum mechanics that can be solved using numerical methods. Python libraries such as numpy, scipy, and matplotlib provide powerful tools to solve the Schrödinger equation numerically, visualize the results, and analyze the data.
# 
# The objectives for using Python for computational physics may include:
# 
# Solving complex physical problems that cannot be solved analytically using numerical methods.
# Studying physical systems that are too large, too small, too complex, or too dangerous to be studied experimentally.
# Developing and implementing numerical algorithms for simulating physical systems.
# Visualizing and analyzing the results of numerical simulations.
# Comparing numerical simulations with experimental data to test theoretical models and improve our understanding of physical systems.
# Communicating the results of computational physics research to a wider audience.

# # Method and Results

# In[66]:



import numpy as np
import matplotlib.pyplot as plt



def schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs = 20, findpsi = False):

    #Solves the 1 dimensional Schrodinger equation numerically.
    
    x = np.linspace(xmin, xmax, Nx)  # x axis grid.
    dx = x[1] - x[0]  # x axis step size.

    # Obtain the potential function values:
    V = Vfun(x, params)

    # Create the Hamiltonian matrix:
    H = sparse.eye(Nx, Nx, format = "lil") * 2
    for i in range(Nx - 1):
        #H[i, i] = 2
        H[i, i + 1] = -1
        H[i + 1, i] = -1
    #H[-1, -1] = 2
    H = H / (dx ** 2)

    # Add the potential into the Hamiltonian
    for i in range(Nx):
        H[i, i] = H[i, i] + V[i]
    # convert to csc matrix format
    H = H.tocsc()
    
    # Obtain neigs solutions from the sparse matrix
    [evl, evt] = sla.eigs(H, k = neigs, which = "SM")

    for i in range(neigs):
        # Normalize the eigen vectors.
        evt[:, i] = evt[:, i] / np.sqrt(
                                np.trapz(np.conj(
                                evt[:, i]) * evt[:, i], x))
        # Eigen values MUST be real:
        evl = np.real(evl)
    if findpsi == False:
        return evl
    else: 
        return evl, evt, x


def eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs, findpsi = True):

  #  Evaluates the wavefunctions given a particular potential energy function Vfun.

    H = schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs, findpsi)
    evl = H[0] # Energy eigen values.
    indices = np.argsort(evl)
    
    print("Energy eigenvalues:")
    for i,j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, j))
        
    evt = H[1] # eigen vectors 
    x = H[2] # x dimensions 
    i = 0
    
    plt.figure(figsize = (8, 8))
    while i < neigs:
        n = indices[i]
        y = np.real(np.conj(evt[:, n]) * evt[:, n])  
        plt.subplot(neigs, 1, i+1)  
        plt.plot(x, y)
        plt.axis('off')
        i = i + 1  
    plt.show()


def sho_wavefunctions_plot(xmin = -10, xmax = 10, Nx = 500, neigs = 20, params = [1]):

    # Plots the 1D quantum harmonic oscillator wavefunctions.
    
    def Vfun(x, params):
        V = params[0] * x**2
        return V
    
    eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs, True)
    

def infinite_well_wavefunctions_plot(xmin = -10, xmax = 10, Nx = 500, neigs = 20, params = 1e10):
    
  #  Plots the 1D infinite well wavefunctions.
    
    def Vfun(x, params):
        V = x * 0
        V[:100]=params
        V[-100:]=params
        return V
    
    eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs, True)
    
    
def double_well_wavefunctions_plot(xmin = -10, xmax = 10, Nx = 500, neigs = 20, params = [-0.5, 0.01, 7]):
   
   # Plots the 1D double well wavefunctions.
   

    def Vfun(x, params):
        A = params[0]
        B = params[1]
        C = params[2]
        V = A * x ** 2 + B * x ** 4 + C
        return V

    eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs, True)


# In[67]:


sho_wavefunctions_plot()


# When the function sho_wavefunctions_plot is used, the quantum harmonic oscillator will have its first 20 probabilities plotted in order of rising energy states. We can see that the particle is most likely to be located at the center of the potential well, where the potential energy is lowest, for the lowest energy level at the top of the plot. The likelihood that a particle will be discovered elsewhere rises as energy levels rise since the particle can ascend to areas on either side of the well's center with higher potential energy.
# 
# Additionally, especially for the smaller values of n, the numerically calculated energy eigenvalues for the 20 discrete energy levels n turn out to closely match the theoretical values provided by E = 2n + 1, indicating the success of our 1 dimensional Schrödinger equation solver.

# # Conclution

# This completes the numerical solution of the time-independent Schrödinger equation in one dimension.This show how physicists use computational techniques to understand quantum mechanics.
# 
# One-dimensional systems are rarely very realistic, and in order to simulate many physical systems accurately, we need to reach at least two dimensions. with this strategy we can make the Python functions more effective at solving the Schrödinger equation in two or three dimensions.
# 
# Coming from C++ and Java class from 2nd and 3rd year respectively Python to me has peven to be more simple and easier to work with.And i would like to study more about python so it could become my primary progamming language.
# 

# # Recommendation
# 

# I would like to recommend introdution to  Machine Learning since it has a promissing tool for scientist to analize scientific data.

# In[ ]:




