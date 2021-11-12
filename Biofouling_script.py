# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:24:22 2021

@author: Marleen van Soest and Sophie van Mil
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# saveloc = 'C:\\Users\\marle\\OneDrive\\Documenten\\Msc_CP\\SOAC\BioFouling\\'
saveloc = '/Users/sophievanmil/Documents/Climate_Physics/SOAC/Project_Biofouling/'


# Constants and parameters
g = 9.81 # [m/s^2]
nu = 0.00109 # [N s/m^2], dynamic viscosity
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

drag = True
vary = 'vary_shapes' # vary_shapes or vary_densities or vary_sizes
many_densities = False
many_sizes = False

shapes = 4

rho_pl = np.array([50, 500, 950, 1020])
if vary == 'vary_densities' and many_densities == True:
    rho_pl = np.array([50, 280, 285, 870, 875, 1020])

densities = len(rho_pl)

radius = np.array([0.0001, 0.001, 0.005, 0.01]) #m
if vary == 'vary_sizes' and many_sizes == True:
    radius = np.linspace(0.0001,0.02, 100) #m
  
sizes = len(radius)

# Plotting
colors_shapes = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e'] 
colors_densities = [ '#1f77b4', '#ff7f0e','#d62728', '#2ca02c'] 
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams.update({"axes.facecolor":(1,1,1,0.7),"savefig.facecolor":(1,1,1,0.7)})


#%% VARY SHAPES/DENSITY/RADIUS
if vary == 'vary_shapes':
    theta_p = np.zeros(shapes)
    A_proj = np.zeros(shapes)
    
    rho_pl = 950 # [kg/m^3], plastic density
  
    # sphere
    R_s = 0.001 # [m], radius of particle
    theta_p[0] = 4*np.pi*R_s**2 # [m^2], surface of particle
    V_pl = np.ones(shapes)*4/3*np.pi*R_s**3 # [m^3], plastic particle volume
    A_proj[0] = np.pi * R_s**2
    
    # cylinder falling vertically
    R_c = (V_pl[0]/(np.pi*200))**(1/3) # [m], radius
    L_c = 200*R_c # [m], length of cylinder
    theta_p[1] = 2*np.pi*R_c*L_c + 2*np.pi*R_c**2
    A_proj[1] = np.pi*R_c**2
    
    # cylinder falling horizontally
    R_c = (V_pl[0]/(np.pi*200))**(1/3) # [m], radius
    L_c = 200*R_c # [m], length of cylinder
    theta_p[2] = 2*np.pi*R_c*L_c + 2*np.pi*R_c**2
    A_proj[2] = L_c * 2 * R_c
    
    # film (square)
    D_fi = (V_pl[0]/10000)**(1/3)
    L_fi = 100*D_fi
    theta_p[3] = 2*L_fi**2 + 4*L_fi*D_fi
    A_proj[3] = L_fi**2
      
    d_s = 2 * np.sqrt(1/(4*np.pi)*theta_p)
    d_n = 2 * np.sqrt(A_proj/np.pi)   
    d_v = 2*R_s
    K = (1/3)*(d_n/d_v)+(2/3)*(d_s/d_v)

elif vary == 'vary_densities':
    R_s = 0.001 # [m], radius of particle
    theta_p = np.ones(densities)*4*np.pi*R_s**2 # [m^2], surface of particle
    V_pl = np.ones(densities)*4/3*np.pi*R_s**3 # [m^3], plastic particle volume
    
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

# I_fluc = np.ones(len(t))*I_0

I = np.matmul(I_fluc.reshape((len(I_fluc),1)),np.exp(epsilon * z).reshape((1,len(z))))

# Algae growth
T_min  = 0.2                        # Minimum temperature for algae growth [degrees Celcius]
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
z_25 = z_p[int(25*24*3600/dt):int(26*24*3600/dt),:]
rho_p = np.zeros((len_t,length))
A = np.zeros((len_t,length))
V_bf = np.zeros((len_t,length))
V_p = np.zeros((len_t,length))
nr_peaks = np.zeros(length)


# initial conditions
V_bf[0,0] = V_pl[0]/100
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
            V_p[i+1,j] = V_bf[i+1,j] + V_pl[0]
            rho_p[i+1,j] = (V_bf[i+1,j]*rho_bf + V_pl[j]*rho_pl[j])/(V_p[i+1,j])        
        elif vary == 'vary_shapes':
            V_p[i+1,j] = V_bf[i+1,j] + V_pl[0]
            rho_p[i+1,j] = (V_bf[i+1,j]*rho_bf + V_pl[j]*rho_pl)/(V_p[i+1,j])
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
            
    z_25[:,j] = z_p[int(25*24*3600/dt):int(26*24*3600/dt),j]
    
    nr_peaks[j] = len(find_peaks(-z_25[:,j])[0])
    
#%%
## Particle depth

if vary == 'vary_shapes' and drag == True:  
    plt.figure(figsize=(8,6)) 
    plt.plot(t/(3600)-(25*24), z_p[:,0], color = colors_shapes[0], label='Sphere, K = {0:1.3f}'.format(K[0]))
    plt.plot(t/(3600)-(25*24), z_p[:,1], color = colors_shapes[1], label='Cylinder: vertical, K = {0:1.3f}'.format(K[1]))
    plt.plot(t/(3600)-(25*24), z_p[:,2], color = colors_shapes[2], label='Cylinder: horizontal, K = {0:1.3f}'.format(K[2]))
    plt.plot(t/(3600)-(25*24), z_p[:,3], color = colors_shapes[3], label='Film, K = {0:1.3f}'.format(K[3]))
    plt.xlim([0,48])
    plt.xticks(np.arange(0,54,6))
    plt.xlabel('Time [hours]')
    plt.ylabel('Depth [m]')
    plt.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),
          ncol=2, fancybox=True)
    #plt.title('Plastic particle oscillation for various shapes with equal volume, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
    plt.savefig(saveloc+"shapes_dep.png",bbox_inches='tight')
    plt.show()

    
    # Initial results - only sphere of 1 mm, full time
    plt.figure(figsize=(8,6)) 
    plt.plot(t/(3600*24), z_p[:,0], color = colors_shapes[0], label='Sphere')
    plt.xlim([0,30])
    # plt.xticks(np.arange(0,54,6))
    plt.xlabel('Time [days]')
    plt.ylabel('Depth [m]')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15), ncol=2, fancybox=True)
    plt.legend()
    #plt.title('Plastic particle oscillation for various shapes with equal volume, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
    plt.savefig(saveloc+"sphere_totaltime.png",bbox_inches='tight')
    plt.show()
    
    ## Biofilm volume
    plt.figure(figsize=(8,6))
    plt.plot(t/(3600*24), V_bf[:,0], color = colors_shapes[0], label='Sphere')
    plt.plot(t/(3600*24), V_bf[:,1], color = colors_shapes[1], label='Cylinder falling vertically')
    plt.plot(t/(3600*24), V_bf[:,2], color = colors_shapes[2], label='Cylinder falling horizontally')
    plt.plot(t/(3600*24), V_bf[:,3], color = colors_shapes[3], label='Film')
    plt.xlabel('Time [days]')
    plt.ylabel('Biofilm volume [m$^{3}]$')
    plt.legend()
    #plt.title('Effect of shape on biofilm volume')
    plt.savefig(saveloc+"shapes_bf.png",bbox_inches='tight')
    plt.show()
    
    # Algae per m^2
    plt.figure(figsize=(8,6))
    plt.plot(t/(3600*24), A[:,0], color = colors_shapes[0], label='Sphere')
    plt.plot(t/(3600*24), A[:,1], color = colors_shapes[1], label='Cylinder falling vertically')
    plt.plot(t/(3600*24), A[:,2], color = colors_shapes[2], label='Cylinder falling horizontally')
    plt.plot(t/(3600*24), A[:,3], color = colors_shapes[3], label='Film')
    plt.xlabel('Time [days]')
    plt.ylabel('A [Nr algae /m$^2$] ')
    plt.legend()
    #plt.title('Effect of shape on algae thickness')
    plt.savefig(saveloc+"shapes_A.png",bbox_inches='tight')
    plt.show()
    
    w_max = np.max(w, axis=0)
    z_max = np.min(z_25, axis=0)

if vary == 'vary_densities':    
    if many_densities == False:
        plt.figure(figsize=(8,6)) 
        for i in np.arange(0,len(rho_pl),1):
            plt.plot(t/(3600)-(24*25), z_p[:,i], label='Density = {} kg/m$^3$'.format(rho_pl[i]))
        plt.xlim([0,24])
        plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
        plt.xlabel('Time [hours]')
        plt.ylabel('Depth [m]')
        plt.legend()
        #plt.title('Plastic particle oscillation for spheres of various densities, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
        plt.savefig(saveloc+"densities_dep.png",bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(8,6)) 
        plt.plot(t/(3600*24), z_p[:,-1], 'r', label='Density = {} kg/m$^3$'.format(rho_pl[-1]))
        # plt.xlim([0,30])
        plt.xticks([0, 5, 10, 15, 20, 25, 30])
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.legend()
        #plt.title('Plastic particle oscillation for density similar to seawater') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
        plt.savefig(saveloc+"densities_long.png",bbox_inches='tight')
        # plt.show()
        
    if many_densities == True:
        # FIND AMOUNT OF OSCILLATIONS
        plt.figure(figsize=(8,6)) 
        plt.plot(rho_pl, nr_peaks)
        plt.xlabel('Plastic density [kg/m$^3$]')
        plt.yticks([0, int(1), int(2), int(3)])
        plt.ylabel('Nr oscillations')
        # plt.title('Plastic particle oscillation for density similar to seawater') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
        plt.savefig(saveloc+"densities_osc.png",bbox_inches='tight')
        plt.show()

if vary == 'vary_sizes':
     if many_sizes == False:
         plt.figure(figsize=(8,6)) 
         for i in range(sizes):
             plt.plot(t/(3600)-25*24, z_p[:,i], label='Radius = {} mm'.format(R_s[i]*1000))
         plt.xlim([0, 24])
         plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
         plt.xlabel('Time [hours]')
         plt.ylabel('Depth [m]')
         plt.legend()
        # plt.title('Plastic particle oscillation for spheres of various sizes, on day 25') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
         plt.savefig(saveloc+"sizes_dep.png",bbox_inches='tight')
         plt.show()
        
         # Plot of smallest size (0.1 mm), for total time interval
         plt.figure(figsize=(8,6)) 
         plt.plot(t/(3600*24), z_p[:,0], label='Radius = {} mm'.format(R_s[0]*1000))
         plt.xlabel('Time [days]')
         plt.ylabel('Depth [m]')
         plt.legend()
         #plt.title('Plastic particle oscillation for smallest sphere') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
         plt.savefig(saveloc+"sizes_small.png",bbox_inches='tight')
         plt.show()
         
     if many_sizes == True:
        # max velocity vs size
        w_max = np.max(w, axis=0)
        
        plt.figure(figsize=(8,6)) 
        plt.plot(radius*1000, w_max, '.')
        # plt.xlim([0,20])
        plt.xlabel('Particle radius [mm]')
        plt.ylabel('Max velocity [m/s]')
        # plt.title('Plastic particle oscillation for density similar to seawater') #', shape-dependant drag = '+ str(drag)) #', R = '+ str(R) + ' m')
        plt.savefig(saveloc+"max_vel.png",bbox_inches='tight')
        plt.show()

#%% Profiles

## Seawater light fluctuation
plt.figure(figsize=(8,6))
plt.contourf(t/(3600)-25*24,z,np.transpose(I))
plt.xlim([0,24])
plt.ylim([-20,0])
plt.xlabel('Time [hours]')
plt.ylabel('Depth [m]')
plt.colorbar(label='Light intensity [$\mu$ E /(m$^{2}$s)]')
#plt.title('Light penetration')
plt.savefig(saveloc+"sealight_zoomed.png",bbox_inches='tight')
plt.show()

## Seawater light fluctuation
plt.figure(figsize=(8,6))
plt.contourf(t/(3600*24),z,np.transpose(I))
plt.xlim([25,30])
plt.ylim([-20,0])
plt.xlabel('Time [days]')
plt.ylabel('Depth [m]')
plt.colorbar(label='Light intensity [$\mu$ E /(m$^{2}$s)]')
#plt.title('Light penetration')
plt.savefig(saveloc+"sealight.png",bbox_inches='tight')
plt.show()

# Seawater temperature
plt.figure(figsize=(8,6))
plt.plot(T, z, 'r')
plt.xlabel('Temperature [$\degree$ C]')
plt.ylabel('Depth [m]')
plt.ylim([-1000,0])
#plt.title('Temperature profile of the North-Pacific ocean')
plt.savefig(saveloc+"temp_NP.png",bbox_inches='tight')
plt.show()

# Seawater salinity
plt.figure(figsize=(8,6))
plt.plot(S*1000, z, color='grey')
plt.xlabel('Salinity [g/kg]')
plt.ylabel('Depth [m]')
plt.ylim([-1000,0])
#plt.title('Salinity profile of the North-Pacific ocean')
plt.savefig(saveloc+"salinity_NP.png",bbox_inches='tight')
plt.show()

#%%
fig,ax = plt.subplots(figsize=(4,9))
# make a plot
ax.plot(T, z, 'r')
ax.set_xlabel("Temperature [$\degree$ C]",color='red')
ax.tick_params(axis='x', colors='red')
ax.set_ylabel("Depth [m]")
ax2=ax.twiny()
ax2.plot(S*1000, z, color='grey')
ax2.tick_params(axis='x', colors='grey')
ax2.set_xlabel("Salinity [g/kg]",color="grey")
ax.set_ylim([-1000,0])
fig.savefig(saveloc+"temp_sal.png", bbox_inches='tight')
plt.show()
  		
# Seawater density
plt.figure(figsize=(4,9))
plt.plot(rho_sw, z, color='navy')
plt.xlabel('Seawater density [kg/m$^3$]')
plt.ylabel('Depth [m]')
plt.title(' ')
plt.xticks([1024.0, 1025.0, 1026.0, 1027.0])
plt.ylim([-1000,0])
#plt.title('Density profile of the North-Pacific ocean')
plt.savefig(saveloc+"density_NP.png",bbox_inches='tight')
plt.show()

#%% Animation preparation
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib as mpl 


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\marle\\OneDrive\\Documenten\\ffmpeg\\bin\\ffmpeg.exe'


#%% Shapes animation
if vary == 'vary_shapes' and drag == True:  
    fig, ax = plt.subplots(figsize=(10, 16))
    fig.set_tight_layout(True)
    ax.set(xlim=(-0.2,3.2), ylim=(-100, 0))
    # ax.set_ylabel('Depth', fontsize=20)
    
    x = np.array([0,1,2,3])
    z_5 = z_p[int(25*24*3600/dt):-1,:]
    t_5 = t[int(25*24*3600/dt):-1]/(3600)-(24*25)
    day = t_5/24+25
    t_5 = t[int(25*24*3600/dt):-1]/(3600)-(24*np.floor(day))
    
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = ['Sphere', 'Cylinder (hor)', 'Cylinder (ver)', 'Film']
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ylabels = ['Surface (z=0)']
    markers = np.repeat(['o','|','_','s'], len(xlabels)/4)
    linewidths= [1,3,3,1]
    sizes = [200, 1000, 1000, 200]
    
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=20)
    ax.set_yticks(np.array([0]))
    ax.set_yticklabels(ylabels, fontsize=20)
    
    scat = mscatter(x, z_5[0,:], c=colors_shapes, s=sizes, m=markers, linewidths=5, ax=ax)
    
    def animate(i):
        y = z_5[i,:]
        scat.set_offsets(np.c_[x, y]) 
        ax.set_title('day '+str(int(np.floor(day[i])))+', hour '+str(int(np.floor(t_5[i]))), fontsize=20)
    
    anim = FuncAnimation(fig, animate, interval=100, repeat=True, save_count=len(z_5[:,0])-1)

writervideo = animation.FFMpegWriter(fps=100) 
anim.save(saveloc+'animation_shape.mp4', writer=writervideo)

#%% Densities animation
if vary == 'vary_densities':    
    fig, ax = plt.subplots(figsize=(10, 16))
    fig.set_tight_layout(True)
    ax.set(xlim=(-0.5,3.5), ylim=(-100, 0))
    # ax.set_ylabel('Depth', fontsize=20)
    
    x = np.array([0,1,2,3])
    z_5 = z_p[int(25*24*3600/dt):-1,:]
    t_5 = t[int(25*24*3600/dt):-1]/(3600)-(24*25)
    day = t_5/24+25
    t_5 = t[int(25*24*3600/dt):-1]/(3600)-(24*np.floor(day))
    
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = ['50 kg/m$^3$', '500 kg/m$^3$', '950 kg/m$^3$', '1020 kg/m$^3$']
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ylabels = ['Surface (z=0)']
    markers = np.repeat(['o','|','_','s'], len(xlabels)/4)
    linewidths= [1,3,3,1]
    sizes = [200, 1000, 1000, 200]
    
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=20)
    ax.set_yticks(np.array([0]))
    ax.set_yticklabels(ylabels, fontsize=20)
    
    scat = mscatter(x, z_5[0,:], color=colors_densities, s=300, ax=ax)
    
    def animate(i):
        y = z_5[i,:]
        scat.set_offsets(np.c_[x, y]) 
        ax.set_title('day '+str(int(np.floor(day[i])))+', hour '+str(int(np.floor(t_5[i]))), fontsize=20)
    
    anim = FuncAnimation(fig, animate, interval=100, repeat=True, save_count=len(z_5[:,0])-1)
    
    writervideo = animation.FFMpegWriter(fps=100) 
    anim.save(saveloc+'animation_dens.mp4', writer=writervideo)

#%% Size animation
if vary == 'vary_sizes':    
    fig, ax = plt.subplots(figsize=(10, 16))
    fig.set_tight_layout(True)
    ax.set(xlim=(-0.5,3.5), ylim=(-100, 0))
    # ax.set_ylabel('Depth', fontsize=20)
    
    x = np.array([0,1,2,3])
    z_5 = z_p[int(25*24*3600/dt):-1,:]
    t_5 = t[int(25*24*3600/dt):-1]/(3600)-(24*25)
    day = t_5/24+25
    t_5 = t[int(25*24*3600/dt):-1]/(3600)-(24*np.floor(day))
    
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    xlabels = ['$R_{pl}$ = 0.1 mm', '1 mm', '5 mm', '10 mm']
    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    ylabels = ['Surface (z=0)']
    markers = np.repeat(['o','|','_','s'], len(xlabels)/4)
    linewidths= [1,3,3,1]
    sizes = [200, 300, 400, 500]
    
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=20)
    ax.set_yticks(np.array([0]))
    ax.set_yticklabels(ylabels, fontsize=20)
    
    scat = mscatter(x, z_25[0,:], color=colors_densities, s=sizes, ax=ax)
    
    def animate(i):
        y = z_5[i,:]
        scat.set_offsets(np.c_[x, y]) 
        ax.set_title('day '+str(int(np.floor(day[i])))+', hour '+str(int(np.floor(t_5[i]))), fontsize=20)
    
    anim = FuncAnimation(fig, animate, interval=100, repeat=True, save_count=len(z_5[:,0])-1)
    
    writervideo = animation.FFMpegWriter(fps=100) 
    anim.save(saveloc+'animation_size.mp4', writer=writervideo)
