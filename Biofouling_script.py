# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:24:22 2021

@author: Marleen van Soest and Sophie van Mil
"""

import numpy as np
import matplotlib.pyplot as plt


# Constants and parameters
g = 9.81 # [m/s^2]
nu = 0.00109 # [N s/m^2], viscosity
rho_bf = 1388 # [kg/m^3], algae density
m_A = 0.39/(3600*24) # [/s], mortality rate
R_20 = 0.1/(3600*24) # [/s], respiration rate
Q_10 = 2 # [-], temperature coeff respiration
V_A = 2*10**(-16) # [m^3], volume per algae

# Model settings
dt = 60
nr_days = 30
len_t = int(nr_days*24*3600/dt)
t = np.arange(0,len_t*dt, dt)

D = 999 # [m], depth of grid
dz = 1 # [m]
z = np.arange(-D, 1, dz)
z = np.flip(z)

drag = False
vary = 'vary_shapes' # vary_shapes or vary_densities or vary_sizes

shapes = 4

rho_pl = np.array([50, 500, 950])
densities = len(rho_pl)

radius = np.array([0.0001, 0.001, 0.005, 0.01]) #m
sizes = len(radius)

# Plotting
colors_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

#%% VARY SHAPES/DENSITY/RADIUS

if vary == 'vary_shapes':
    theta_p = np.zeros(shapes)
    d_s = np.zeros(shapes)
    d_n = np.zeros(shapes)
    length_ratio = np.zeros(shapes)
    axis_ratio = np.zeros(shapes)
    K = np.zeros(shapes)
    
    rho_pl = 950 # [kg/m^3], plastic density
  
    # sphere
    R_s = 0.001 # [m], radius of particle
    theta_p[0] = 4*np.pi*R_s**2 # [m^2], surface of particle
    V_pl = 4/3*np.pi*R_s**3 # [m^3], plastic particle volume
    
    d_s[0] = 2*R_s
    d_n[0] = 2*R_s
    length_ratio[0] = (2*R_s)**2/(np.pi*R_s**2)
    axis_ratio[0] = 1
    K[0] = 0.357 + 0.684*(d_s[0]/d_n[0]) + 0.00154*length_ratio[0] + 0.0104*axis_ratio[0]
    
    # cylinder falling vertically
    R_c = (V_pl/(np.pi*200))**(1/3) # [m], radius
    L_c = 200*R_c # [m], length of cylinder
    theta_p[1] = 2*np.pi*R_c*L_c + 2*np.pi*R_c**2
    
    d_s[1] = np.sqrt(2*(R_c*L_c+R_c**2))
    d_n[1] = 2*R_c
    length_ratio[1] = L_c**2/(np.pi*R_c**2)
    axis_ratio[1] = 1
    K[1] = 0.357 + 0.684*(d_s[1]/d_n[1]) + 0.00154*length_ratio[1] + 0.0104*axis_ratio[1]
    
    # cylinder falling horizontally
    R_c = (V_pl/(np.pi*200))**(1/3) # [m], radius
    L_c = 200*R_c # [m], length of cylinder
    theta_p[2] = 2*np.pi*R_c*L_c + 2*np.pi*R_c**2
    
    d_s[2] = d_s[1] # np.sqrt(2*(R_c*L_c+R_c**2))
    d_n[2] = np.sqrt(8*R_c*L_c/np.pi)
    length_ratio[2] = 2*R_c/L_c
    axis_ratio[2] = L_c/(2*R_c)
    K[2] = 0.357 + 0.684*(d_s[2]/d_n[2]) + 0.00154*length_ratio[2] + 0.0104*axis_ratio[2]
    
    # film (square)
    D_fi = (V_pl/10000)**(1/3)
    L_fi = 100*D_fi
    theta_p[3] = 2*L_fi**2 + 4*L_fi*D_fi
    
    d_s[3] = 2*np.sqrt(1/(2*np.pi)*(L_fi**2+2*L_fi*D_fi))
    d_n[3] = 2*np.sqrt(L_fi**2/np.pi)
    length_ratio[3] = D_fi**2/(L_fi**2)
    axis_ratio[3] = 1
    K[3] = 0.357 + 0.684*(d_s[3]/d_n[3]) + 0.00154*length_ratio[3] + 0.0104*axis_ratio[3]
    
    K = K * d_n/(2*R_s)     # before we actually had Kn, not K

elif vary == 'vary_densities':
    R_s = 0.001 # [m], radius of particle
    theta_p = np.ones(densities)*4*np.pi*R_s**2 # [m^2], surface of particle
    V_pl = 4/3*np.pi*R_s**3 # [m^3], plastic particle volume
    
    K = np.ones(densities)

elif vary == 'vary_sizes':
    rho_pl = 950 # [kg/m^3], plastic density
    R_s = radius # [m], radius of particle
    theta_p = rho_pl*4*np.pi*R_s**2 # [m^2], surface of particle
    V_pl = 4/3*np.pi*R_s**3 # [m^3], plastic particle volume
    
    K = np.ones(sizes)
    
#%% Profiles

# Temperature profile
T_surf = 25     # Water temperature at the surface [degrees Celcius], for the North Pacific
T_bot  = 1.5    # Water temperature at the sea bottom [degrees Celcius], for the North Pacific
z_c    = -300   # Depth of the thermocline [m], for the North Pacific
p      =  2     # Parameter defining the steepness of the thermocline, for the North Pacific

T = T_surf + (T_bot - T_surf)*(z**p)/(z**p + z_c**p)

#Salinity profile - North Pacific
c1, c2, c3, c4, c5, c6 = 9.998E-17, 1.054E-12, 3.997E-9, 6.541E-6, 4.195E-3, 3.517E+1   
S = (c1*z**5 + c2*z**4 + c3*z**3 + c4*z**2 + c5*z + c6)/1000

# Density profile - North Pacific
a1, a2, a3, a4, a5 = 9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8
b1, b2, b3, b4, b5 = 8.020E2, -2.001, 1.677E-2, 2.261E-5, -4.657E-5 
rho_sw = (a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4) + (b1*S + b2*S*T + b3*S*T**2 + b4*S*T**3 + b5*S**2*T**2)

# Light profile
epsilon = 0.2
I_0 = 1.2e8 /(24*3600) # [mu E /m^2/s], light intensity at surface without day/night

I_fluc = I_0 * np.sin(2*np.pi*(t-6*(3600))/(3600*24))
I_fluc = np.where(I_fluc < 0, 0, I_fluc)

I = np.matmul(I_fluc.reshape((len(I_fluc),1)),np.exp(epsilon * z).reshape((1,len(z))))

# Algae growth
T_min  = 0.2                        # Minimum temperatuer for algae growth [degrees Celcius]
T_max  = 33.3                       # Maximum temperature for algae growth [degrees Celcius]
T_opt  = 26.7                 
mu_max = 1.85/(3600*24)             # Maximum growth rate algae [per second]
I_opt  = 1.75392*10**13/(3600*24)   # Optimal light intensity algae growth [micro Einstein per m2 per second]
alpha  = 0.12/(3600*24)             # Initial slope [per second]

mu_opt = mu_max * I / (I + (mu_max/alpha) * ((I/I_opt)-1)**2)
phi = ((T-T_max)*(T-T_min)**2)/((T_opt-T_min)*((T_opt-T_min)*(T-T_opt)-(T_opt-T_max)*(T_opt+T_min-2*T)))
mu = mu_opt*phi

indexmin = np.argwhere(T < T_min)
indexmax = np.argwhere(T > T_max)
mu[indexmin] = 0
mu[indexmax] = 0

#%% Integration

if vary == 'vary_shapes':
    length = shapes
elif vary == 'vary_densities':
    length = densities
elif vary == 'vary_sizes':
    length = sizes
  
# define matrices
w = np.zeros((len_t,length))
z_p = np.zeros((len_t,length))
rho_p = np.zeros((len_t,length))
A = np.zeros((len_t,length))
V_bf = np.zeros((len_t,length))
V_p = np.zeros((len_t,length))

# initial conditions
V_bf[0,0] = V_pl/100
A[0,0] = V_bf[0,0]/(V_A*theta_p[0])
A[0,:] = A[0,0]
V_bf[0,:] = V_A*A[0,:]*theta_p[:]

V_p[0,:] = V_bf[0,:] + V_pl
w[0,:] = 0
z_p[0,:] = 0
rho_p[0,:] = (V_bf[0,:]*rho_bf + V_pl*rho_pl)/(V_p[0,:])

# integrate
for j in range(length):
    for i in range(len_t-1):
        index_p = abs(z - z_p[i,j]).argmin()
        
        dAdt = mu[int(t[i]/dt),index_p]*A[i,j] - m_A * A[i,j] - Q_10**((T[index_p]-20)/10) * R_20 * A[i,j]
        A[i+1,j] = A[i,j] + dAdt*dt
        
        V_bf[i+1,j] = V_A*A[i+1,j]*theta_p[j]
    
        if vary == 'vary_densities': 
            V_p[i+1,j] = V_bf[i+1,j] + V_pl
            rho_p[i+1,j] = (V_bf[i+1,j]*rho_bf + V_pl*rho_pl[j])/(V_p[i+1,j])        
        elif vary == 'vary_shapes':
            V_p[i+1,j] = V_bf[i+1,j] + V_pl
            rho_p[i+1,j] = (V_bf[i+1,j]*rho_bf + V_pl*rho_pl)/(V_p[i+1,j])
        elif vary == 'vary_sizes':
            V_p[i+1,j] = V_bf[i+1,j] + V_pl[j]
            rho_p[i+1,j] = (V_bf[i+1,j]*rho_bf + V_pl[j]*rho_pl)/(V_p[i+1,j])
        
        R_p = (V_p[i+1,j]*(3/4)/np.pi)**(1/3) # equivalent sphere radius
        
        if drag == True:            # Including shape dependant drag
           w[i+1,j] = -(2/9)*(rho_p[i+1,j]-rho_sw[index_p])/(nu*K[j])*g*R_p**2 # terminal velocity (Fg, Fb, Fd)
        else:                       # Drag independant of shape --> all drag as if they were spherical
            w[i+1,j] = -(2/9)*(rho_p[i+1,j]-rho_sw[index_p])/(nu)*g*R_p**2 # terminal velocity (Fg, Fb, Fd)
        
        z_p[i+1,j] = z_p[i,j] + w[i+1,j]*dt
        
        if z_p[i+1,j] >= 0: # particle is at surface
            z_p[i+1,j] = 0
            w[i+1,j] = 0

#%% Figures

if vary == 'vary_shapes' and drag == False:
    plt.figure(figsize=(8,6)) 
    plt.plot(t/(3600)-25*24, z_p[:,0], color = colors_4[3], label='Sphere')
    # plt.plot(t/(3600)-25*24, z_p[:,1], color = colors_4[2], label='Cylinder falling vertically')
    plt.plot(t/(3600)-25*24, z_p[:,2], color = colors_4[0], label='Cylinder falling horizontally')
    plt.plot(t/(3600)-25*24, z_p[:,3], color = colors_4[1], label='Film')
    # plt.xlim([0,24])
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    plt.xlabel('Time [hours]')
    plt.ylabel('Depth [m]')
    plt.legend()
    plt.title('Plastic particle oscillation for various shapes with equal volume, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
    plt.savefig("part_dep_shapes_constantdrag.png")
    plt.show()
 # TO DO: PLOTJE MET GEDEELDE X-AS en zonkracht   
#%%
## Particle depth
if vary == 'vary_shapes' and drag == True:
    plt.figure(figsize=(8,6)) 
    plt.plot(t/(3600)-25*24, z_p[:,0], color = colors_4[3], label='Sphere, K = {0:1.5f}'.format(K[0]))
    plt.plot(t/(3600)-25*24, z_p[:,1], color = colors_4[2], label='Cylinder falling vertically, K = {0:1.5f}'.format(K[1]))
    plt.plot(t/(3600)-25*24, z_p[:,2], color = colors_4[0], label='Cylinder falling horizontally, K = {0:1.5f}'.format(K[2]))
    plt.plot(t/(3600)-25*24, z_p[:,3], color = colors_4[1], label='Film, K = {0:1.5f}'.format(K[3]))
    plt.xlim([0,24])
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    plt.xlabel('Time [hours]')
    plt.ylabel('Depth [m]')
    plt.legend()
    plt.title('Plastic particle oscillation for various shapes with equal volume, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
    plt.savefig("part_dep_shapes.png")
    plt.show()
 # TO DO: PLOTJE MET GEDEELDE X-AS en zonkracht   
    
if vary == 'vary_densities':
    plt.figure(figsize=(8,6)) 
    for i in range(densities):
        plt.plot(t/(3600)-25*24, z_p[:,i], label='Density = {}'.format(rho_pl[i]))
    plt.xlim([0,24])
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    plt.xlabel('Time [hours]')
    plt.ylabel('Depth [m]')
    plt.legend()
    plt.title('Plastic particle oscillation for spheres of various densities, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
    plt.savefig("part_dep_densities.png")
    plt.show()

if vary == 'vary_sizes':
    plt.figure(figsize=(8,6)) 
    for i in range(sizes):
        plt.plot(t/(3600)-25*24, z_p[:,i], label='Radius = {} mm'.format(R_s[i]*1000))
    plt.xlim([0, 24])
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    plt.xlabel('Time [hours]')
    plt.ylabel('Depth [m]')
    plt.legend()
    plt.title('Plastic particle oscillation for spheres of various sizes, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
    plt.savefig("part_dep_sizes.png")
    plt.show()
    
    # Plot of smallest size (0.1 mm), for total time interval
    plt.figure(figsize=(8,6)) 
    plt.plot(t/(3600*24), z_p[:,0], label='Radius = {} mm'.format(R_s[0]*1000))
    plt.xlabel('Time [days]')
    plt.ylabel('Depth [m]')
    plt.legend()
    plt.title('Plastic particle oscillation for spheres of various sizes, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
    plt.savefig("part_dep_sizes_small.png")
    plt.show()
#%%

## Particle density
plt.figure(figsize=(8,6))
plt.plot(t/(3600*24), V_bf[:,0], label='sphere')
plt.plot(t/(3600*24), V_bf[:,1], label='cylinder')
plt.plot(t/(3600*24), V_bf[:,2], label='cylinder')
plt.plot(t/(3600*24), V_bf[:,3], label='film')
# plt.plot(t/(3600*24), rho_p[:,3], label='foam')
# plt.xlim([25,30])
# plt.ylim([950,np.max(rho_p)+10])
plt.xlabel('time (days)')
plt.ylabel('Biofilm volume [$kg/m^{3}]$')
plt.legend()
plt.title('Plastic particle density during oscillation')
plt.savefig("part_dens.png")
plt.show()

## Particle density
plt.figure(figsize=(8,6))
plt.plot(t/(3600*24), A[:,0], label='sphere')
plt.plot(t/(3600*24), A[:,1], label='cylinder')
# plt.plot(t/(3600*24), V_bf[:,2], label='cylinder')
plt.plot(t/(3600*24), A[:,3], label='film')
# plt.plot(t/(3600*24), rho_p[:,3], label='foam')
# plt.xlim([25,30])
# plt.ylim([950,np.max(rho_p)+10])
plt.xlabel('time (days)')
plt.ylabel('A ')
plt.legend()
plt.title('Plastic particle density during oscillation')
plt.savefig("part_dens.png")
plt.show()
# ## Particle velocity
# plt.figure(figsize=(8,6))
# plt.plot(t/(3600*24), w[:,0], label='sphere')
# plt.plot(t/(3600*24), w[:,1], label='cylinder')
# plt.plot(t/(3600*24), w[:,2], label='film')
# plt.plot(t/(3600*24), w[:,3], label='foam')
# plt.xlim([25,30])
# plt.xlabel('time (days)')
# plt.ylabel('velocity [m/s]')
# plt.legend()
# plt.title('Plastic particle velocity during oscillation')
# plt.savefig("part_vel.png")
# plt.show()

#%%

## Seawater light fluctuation
plt.figure(figsize=(8,6))
plt.contourf(t/(3600*24),z,np.transpose(I))
plt.xlim([25,30])
plt.ylim([-20,0])
plt.xlabel('Time [days]')
plt.ylabel('Depth [m]')
plt.colorbar(label='Light intensity [$\mu E /(m^{2}s)$]')
plt.title('Light penetration')
plt.savefig("sealight.png")
plt.show()

# Seawater temperature
plt.figure(figsize=(8,6))
plt.plot(T, z, 'r')
plt.xlabel('Temperature [$\degree$ C]')
plt.ylabel('Depth [m]')
plt.title('Temperature profile of the North-Pacific ocean')
plt.savefig("temp_NP.png")
plt.show()

# Seawater salinity
plt.figure(figsize=(8,6))
plt.plot(S*1000, z, color='grey')
plt.xlabel('Salinity [g/kg]')
plt.ylabel('Depth [m]')
plt.title('Salinity profile of the North-Pacific ocean')
plt.savefig("salinity_NP.png")
plt.show()
  		
# Seawater density
plt.figure(figsize=(8,6))
plt.plot(rho_sw, z, color='navy')
plt.xlabel('Seawater density [$kg/m^3$]')
plt.ylabel('Depth [m]')
plt.title('Density profile of the North-Pacific ocean')
plt.savefig("density_NP.png")
plt.show()


