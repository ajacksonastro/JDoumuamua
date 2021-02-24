#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:27:03 2020

Author: Alan P. Jackson, ORC-ID: 0000-0003-4393-9520

This code computes all of the calculations and produces all of the figures
presented in the paper
"1I/'Oumuamua as an N2 ice fragment of an exo-Pluto surface:
    I. size and compositional constraints", Jackson & Desch, 2021, JGR: Planets

This code is released under the GNU General Public License v2.0 (GPL v2)
a copy of which should be distributed with all copies of this code.
"""

import numpy as np
from scipy import optimize
from scipy import special
import matplotlib as mpt
mpt.use('pdf')
import matplotlib.pyplot as plt

from labellines import labelLine, labelLines

# Set the figure font size and axis linewidth
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.linewidth':2.0})

#constants
grav = 6.67430e-11 #gravitational constant (m^3 kg^-1 s^-2)
Msun = 1.989e30 #Mass of Sun (kg)
aum  = 1.495e11 #1 au in metres
years= 365.25*24.0*60.0*60.0 #1 year in seconds
days = 24.0*60.0*60.0 #1 day in seconds
kb   = 1.380649e-23 #Boltzmann constant (J/K)
sigma= 5.670374e-8  #Stefan-Boltzmann constant (w m^-2 K^-4)
Na   = 6.022141e23 #Avogadro constant (mol^-1)
Lsun = 3.848e26 #Solar luminosity (W)
amu  = 1.660539e-27 #atomic mass unit (kg)

#'Oumuamua parameters
e_orb = 1.2 #eccentricity
a_orb = -1.273*aum #semi-major axis in au

xi_obl= 0.25 #mean fraction of total surface area exposed to solar radiation for oblate solution (SL2020)

#Axis scale normalisations.
#Absolute size is uncertain and determined by albedo, p.  Surface area scales as 1/p, thus axis scales as 1/sqrt(p)
#Normalisations give 2a = 115 m, 2b = 111 m, and 2c = 19 m for p=0.1 as given by SL2020, Mashchenko 2019
La = 18.18 #scale for a axis of ellipsoid model (m)
Lb = 17.55 #scale for b axis of ellipsoid model (m)
Lc =  3.00 #scale for c axis of ellipsoid model (m)

#Mass of 'Oumuamua, assuming oblate spheroid
def MassObl(La, Lc, dens, p):
    return (4.0/3.0) * np.pi* La**2.0 * Lc *dens / p**1.5

#Mass of 'Oumuamua, triaxial spheroid
def MassTri(La, Lb, Lc, dens, p):
    return (4.0/3.0) * np.pi * La * Lb * Lc * dens / p**1.5

#Surface area of 'Oumuamua, assuming oblate spheroid
def SurfaceAreaObl(La, Lc, p):
    ecc = np.sqrt(1.0 - Lc**2.0/La**2.0)
    return 2.0 * np.pi * (La**2.0 / p) * (1.0 + Lc**2.0 * np.arctanh(ecc) / (ecc * La**2.0) )

#Surface area of 'Oumuamua, triaxial ellipsoid
def SurfaceAreaTri(La, Lb, Lc, p):
    phi = np.arccos(Lc/La)
    k = np.sqrt((La*La*(Lb*Lb - Lc*Lc))/(Lb*Lb*(La*La - Lc*Lc)))
    Fphik = special.ellipkinc(phi, k)
    Ephik = special.ellipeinc(phi, k)
    return 2.0*np.pi*(Lc*Lc/p) + 2.0*np.pi*(La*Lb/p) * (Ephik*np.sin(phi)*np.sin(phi) + Fphik*np.cos(phi)*np.cos(phi))/np.sin(phi)

#Get mean anomaly from hyperbolic anomaly
def HtoM(H, ecc):
    return ecc*np.sinh(H)-H

#function to optimise in HyperKep
def func(H, M, ecc):
    return M-ecc*np.sinh(H)+H

#derivative of func
def fprime(H, M, ecc):
    return 1-ecc*np.cosh(H)

#Solve Kepler's equation and get hyperbolic anomaly from mean anomaly
def HyperKep(M, ecc):
    if ((M > np.pi) or ((M < 0.0) and (M > -np.pi))):
        H0 = M - ecc
    else:
        H0 = M + ecc
        
    res = optimize.newton(func, H0, fprime=fprime, args=(M,ecc), tol=1e-8, maxiter=1000)
    
    return res

class Gas:
    def __init__(self, name, Tsub, mmass, dens, dHsub, dHtrans, cp, thcond, nu, gamma):
        self.name = name #name of gas
        self.Tsub = Tsub #sublimation temperature (K)
        self.mmass = mmass #mean molecular mass (amu)
        self.dens = dens #density (kg m^-3)
        self.dHsub = dHsub #sublimation enthalpy (J/mol)
        self.dHtrans = dHtrans #enthalpy of any solid-solid phase transition that occurs within temperature range (J/mol)
        self.cp = cp #specific heat capacity (J/K/kg)
        self.thcond = thcond #thermal conductivity (w/m/K)
        self.nu = nu #vibrational frequency (s^-1)
        self.gamma = gamma #adiabatic index
        
#Gas data, Tsub, density and deltaH from Shakeel 2018 via SL2020.  Adiabatic index is 5/3 for monatomic gases, 7/5 for diatomic gases, ~1.3 for CO2 and water
#Sources
#N2
#Sublimation temperature: Shakeel et al. 2018
#Density: Shakeel et al. 2018
#Sublimation enthalpy: Mean of values in Shakeel et al. 2018, Fayolle et al. 2016, Oberg et al. 2005, Bisschop et al. 2006, Collings et al. 2015, Frels et al. 1974 gives 6.799 +/- 0.117
#Heat capacity: Trowbridge et al. 2016 (given as 926.91*exp(0.0093T), evaluated at 30K, 1116 at 10K, 1344 at 40K
#Alpha-beta phase transition occurs at 35.66 K with enthalpy 0.215 kJ/mol: Lipinski 2007
#Thermal conductivity: Cook & Davey 1976, Stachowiak et al. 1994 at 20K
#Vibrational frequency: Fayolle et al. 2016 (for 15N, assuming scaling as 1/sqrt(m))
#H2
#Sublimation temperature: Silvera 1980 (vapour pressure = 10^-10.53 Pa at 4K)
#Density: Silvera 1980
#Sublimation enthalpy: Souers 1986 (at 6K)
#Heat capacity: Souers 1986 (at 6 K)
#Thermal conductivity: Huebler & Bohn 1978
#Vibrational frequency: Sandford & Allamandola 1993
#Ne
#sublimation temperature: Shakeel et al. 2018
#Density: Hwang et al. 2005
#sublimation enthalpy: Shakeel et al. 2018
#Heat capacity: Fenichel & Serin 1966 (at 9 K)
#Thermal conductivity: Weston & Daniels 1984 (at 9 K)
#Vibrational frequency: Gupta 1969
#CO2
#Sublimation temperature: Shakeel et al. 2018, Luna et al. 2014
#Density: Shakeel et al. 2018
#Sublimation enthalpy: Mean of values in Luna et al. 2014, Giauque & Egan 1937, Shakeel et al. 2018 (and refs therein) gives 26.5 +/- 2.33 kJ/mol
#Heat capacity: Giaque & Egan 1937 (at 82 K)
#Thermal conductivity: Cook & Davey 1976
#Vibrational frequency: Sandford & Allamandola 1990
#H20
#Sublimation temperature: Shakeel et al. 2018
#Density:
#Sublimation enthalpy: Mean of values in Sandford & Allamandola 1988 and Shakeel et al. 2018 gives 49.58 +/- 5.24
#Heat capacity: Giaque & Stout 1936, Shulman 2004 (at 158 K)
#Thermal conductivity: Slack 1980 (at 150 K)
#Vibrational frequency: Sandford & Allamandola 1988
#O2
#Sublimation temperature: Shakeel et al. 2018
#Density: Shakeel et al. 2018
#Sublimation enthalpy: Shakeel et al. 2018
#Heat capacity: Freiman & Jodl 2004 (and refs therein, at 30 K, beta phase)
#Alpha-Beta phase transition: 90 J/mol at 23.9 K, beta-gamma transition: 750 J/mol at 43.8 K (Szmyrka-Grzebyk et al. 1998, Freiman & Jodl 2004 and refs therein)
#Thermal conductivity: Freiman & Jodl 2004 (and refs therein, at 30 K, beta phase)
#Vibrational frequency: Bier & Jodl 1984 (from Raman peak at 50cm^-1)
#CO
#Sublimation temperature: Luna et al. 2014
#Density: Bierhals 2001 (Ullman's Encyclopedia of Industrial Chemistry)
#Sublimation enthalpy: Luna et al. 2014 (and refs therein)
#Heat capacity: Clayton & Giauque 1932 (at 27 K)
#Thermal conductivity: Stachowiak et al. 1998 (at 20 K)
#Vibrational frequency: Sandford & Allamandola 1988
#CH4
#Sublimation temperature: Luna et al. 2014
#density: Ramsey 1963
#Sublimation enthalpy: Luna et al. 2014 (and refs therein)
#Heat capacity: Vogt & Pitzer 1976 (at 21 K)
#Thermal conductivity: Jezowski et al. 1997 (at 20 K)
#Vibrational frequency: Orbriot et al. 1978 (from Raman peak at 118 cm^-1)
#NH3
#Sublimation temperature: Luna et al. 2014
#Density: Blum 1975
#Sublimation enthalpy: Luna et al. 2014 (and refs therein)
#Heat capacity: Overstreet & Giauque 1937 (at 90 K)
#Thermal conductivity: Romanova et al. 2013 (at 90 K)
#Vibrational frequency: Sandford & Allamandola 1993
H2 = Gas("H$_2$",   4.0,   2.0,   86.0,   850.0, 0.0,    800.0, 1.0, 7.5e12, 1.4)
Ne = Gas("Ne",      9.0,  20.2, 1444.0,  1900.0, 0.0,    207.0, 0.3, 1.4e12, 1.667)
N2 = Gas("N$_2$",  25.0,  28.0, 1020.0,  6799.0, 215.0, 1225.0, 0.3, 6.5e11, 1.4)
O2 = Gas("O$_2$",  31.0,  32.0, 1530.0,  9260.0, 90.0,   870.0, 0.35,1.5e12, 1.4)
CO2= Gas("CO$_2$", 82.0,  44.0, 1560.0, 26500.0, 0.0,    830.0, 0.2, 2.9e12, 1.3)
H2O= Gas("H$_2$O",158.0,  18.0,  920.0, 49580.0, 0.0,   1270.0, 4.3, 2.0e12, 1.3)
CO = Gas("CO",     29.0,  28.0,  930.0,  7300.0, 0.0,    764.0, 0.7, 2.0e12, 1.4)
CH4= Gas("CH$_4$", 36.0,  16.0,  520.0,  9400.0, 0.0,   1203.0, 0.3, 3.5e12, 1.3)
NH3= Gas("NH$_3$",100.0,  17.0,  817.0, 28000.0, 0.0,   1380.0, 4.0, 3.5e12, 1.3)

#error estimate for N2, assume error on dHsub dominates
N2_low = Gas("N$_2$",  25.0,  28.0, 1020.0,  6682.0, 215.0, 1225.0, 0.3, 6.5e11, 1.4)
N2_high= Gas("N$_2$",  25.0,  28.0, 1020.0,  6916.0, 215.0, 1225.0, 0.3, 6.5e11, 1.4)

gasses = [H2, Ne, CH4, N2, CO, O2, NH3, CO2, H2O]

#mass loss rate per unit surface area
def MassLossRate(ice, tsurf):
    mu  = ice.mmass*amu    #mass of 1 molecule in kg
    ndens = ice.dens/mu
    return ice.dens * ice.nu *  np.exp(-ice.dHsub/(Na*kb*tsurf)) / (ndens**(1.0/3.0))

#gas velocity, assuming fluid regime
def GasVel(ice, Tsurf):
    eta=0.45 #Correction factor dependent on Mach number of outflow.  Crifo (1987) recommends ~0.45
    mass=ice.mmass*amu
    return eta*np.sqrt(8.0*kb*Tsurf/(np.pi*mass))

#solve for surface temperature
#determined by energy balance

def Temp_func(Tsurf, ice, p, d):
    mol = ice.mmass/1000.0 #mass of 1 mol in kg
    Tinit = 25.0 #internal temperature of body (K)
    epsilon = 0.85 #IR emissivity of body
    
    subterm    = ice.dHsub/mol #energy to sublimate material (J/kg)
    transterm  = ice.dHtrans/mol #energy of other phase transition (J/kg)
    
    #energy to heat material from internal temperature to surface temperature (J/kg), ignore for very low surface temperatures
    if (Tsurf < Tinit):
        heatterm = 0.0
    else:
        heatterm   = ice.cp*(Tsurf-Tinit)
        
    gasvelterm = 0.5*(GasVel(ice, Tsurf))**2.0 #kinetic energy of escaping gas (J/kg)
    masslossrate = MassLossRate(ice, Tsurf) #rate of mass loss (kg/s/m^2)
    radterm    = epsilon * sigma * Tsurf**4.0 #radiative energy loss (W/m^2), assuming epsilon=1-p
    absterm    = xi_obl * (1.0-p) * Lsun / (4.0*np.pi*d**2.0) #solar energy absorbed (W/m^2)
    
    return (subterm + transterm + heatterm + gasvelterm)*masslossrate + radterm - absterm
    
def Temp_func2(Tsurf, ice, p, d):
    mol = ice.mmass/1000.0 #mass of 1 mol in kg
    Tinit = 25.0 #internal temperature of body (K)
    epsilon = 0.85 #IR emissivity of body
    
    subterm    = ice.dHsub/mol #energy to sublimate material (J/kg)
    transterm  = ice.dHtrans/mol #energy of other phase transition (J/kg)
    
    #energy to heat material from internal temperature to surface temperature (J/kg), ignore for very low surface temperatures
    if (Tsurf < Tinit):
        heatterm = 0.0
    else:
        heatterm   = ice.cp*(Tsurf-Tinit)
        
    gasvelterm = 0.0#0.5*(GasVel(ice, Tsurf))**2.0 #kinetic energy of escaping gas (J/kg)
    masslossrate = MassLossRate(ice, Tsurf) #rate of mass loss (kg/s/m^2)
    radterm    = epsilon * sigma * Tsurf**4.0 #radiative energy loss (W/m^2)
    absterm    = xi_obl * (1.0-p) * Lsun / (4.0*np.pi*d**2.0) #solar energy absorbed (W/m^2)
    
    return (subterm + transterm + heatterm + gasvelterm)*masslossrate + radterm - absterm

#Non-gravitational acceleration
def NonGravAcc(ice, p, Tsurf):
    lambdaf = 1.0/3.0 #geometric factor accounting for variation of sublimation rate with solar zenith angle and orientation of gas outflow
    force = lambdaf*MassLossRate(ice, Tsurf)*GasVel(ice, Tsurf)*SurfaceAreaTri(La, Lb, Lc, p)
    nongravacc = force / MassTri(La, Lb, Lc, ice.dens, p)
    return nongravacc

#Observed non-gravitational acceleration as fit by Micheli 2018
def ObsNonGrav(d):
    return 4.92e-6 * (aum/d)**2.0

def GravAcc(d):
    return grav * Msun / (d**2.0)

p=0.1
d=1.42*aum
Tguess=[6.0, 12.0, 50.0, 50.0, 50.0, 50.0, 130.0, 130.0, 214.0]

pvals = np.linspace(0.0, 1.0, 1000)
Tsurf = np.zeros((9,1000))

for j in np.arange(0,9,1):
    for i in np.arange(0,1000,1):
        res = optimize.root(Temp_func, [Tguess[j]], args=(gasses[j], pvals[i], d), method='hybr')
        Tsurf[j,i] = res.x[0]
    print np.max(Tsurf[j,])

massloss1=np.zeros((9,1000))

for j in np.arange(0,9,1):
    massloss1[j,]=MassLossRate(gasses[j], Tsurf[j,])

Tsurf_N2low = np.zeros(1000)
Tsurf_N2high = np.zeros(1000)

for i in np.arange(0,1000,1):
    res = optimize.root(Temp_func, [50.0], args=(N2_low, pvals[i], d), method='hybr')
    Tsurf_N2low[i] = res.x[0]
    res = optimize.root(Temp_func, [50.0], args=(N2_high, pvals[i], d), method='hybr')
    Tsurf_N2high[i] = res.x[0]

#print np.max(Tsurf_N2low)
#print np.max(Tsurf_N2high)

#set up pyplot figure
fig1=plt.figure(figsize=(8,6), dpi=300)
ax1=fig1.add_subplot(111)
fig1.subplots_adjust(left=0.1, bottom=0.1, right=0.84, top=0.9)

# Set the axis labels
ax1.set_ylabel('surface temperature (K)')
ax1.set_xlabel('albedo')

# Set the axis limits
ax1.set_xlim(0.0,1.0)
ax1.set_ylim(0.0, 250.0)

# Make the tick thickness match the axes
ax1.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax1.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax1.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))
#ax1.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))

colors=['red', 'blue', 'green', 'purple', 'orange', 'brown', 'gold', 'pink', 'turquoise', 'magenta']

for i in np.arange(8,-1,-1):
    Tvspval = ax1.plot(pvals, Tsurf[i,], color=colors[i], linewidth=2.0, label=gasses[i].name)
    #Tvspval2= ax1.plot(pvals, Tsurf2[i,], color='black', linewidth=2.0, linestyle='--', label=gasses[i].name)

xvals = [0.2,0.2,0.4,0.5,0.2,0.33,0.7,0.2,0.4]
labelLines(plt.gca().get_lines(),xvals=xvals)

ax1.legend(bbox_to_anchor=(1.01,1), loc='upper left', frameon=False)

fig1.savefig('Tvsalbedo.pdf')
plt.close(fig1)

#set up pyplot figure
fig2=plt.figure(figsize=(8,6), dpi=300)
ax2=fig2.add_subplot(111)
fig2.subplots_adjust(left=0.1, bottom=0.1, right=0.84, top=0.9)

# Set the axis labels
ax2.set_ylabel('mass loss rate (10$^{-4}$ kg s$^{-1}$ m$^{-2}$')
ax2.set_xlabel('albedo')

# Set the axis limits
ax2.set_xlim(0.0,1.0)
ax2.set_ylim(0.0, 20.0)

# Make the tick thickness match the axes
ax2.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax2.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax2.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))
#ax2.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))

for i in np.arange(8,-1,-1):
    MLvspval = ax2.plot(pvals, massloss1[i,]*1e4, color=colors[i], linewidth=2.0, label=gasses[i].name)
    #MLvspval2= ax2.plot(pvals, massloss2[3,]*1e4, color='black', linewidth=2.0, linestyle='--')
    
#xvals = [0.2,0.2,0.4,0.5,0.2,0.33,0.6,0.2]
#labelLines(plt.gca().get_lines(),xvals=xvals)

ax2.legend(bbox_to_anchor=(1.01,1), loc='upper left', frameon=False)

fig2.savefig('MLvsalbedo.pdf')

plt.close(fig2)

#set up pyplot figure
fig2a=plt.figure(figsize=(8,6), dpi=300)
ax2a=fig2a.add_subplot(111)
fig2a.subplots_adjust(left=0.1, bottom=0.1, right=0.84, top=0.9)

# Set the axis labels
ax2a.set_ylabel('erosion rate (m/day')
ax2a.set_xlabel('albedo')

# Set the axis limits
ax2a.set_xlim(0.0,1.0)
ax2a.set_ylim(0.0, 0.12)

# Make the tick thickness match the axes
ax2a.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax2a.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax2a.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))
#ax2.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))

for i in np.arange(8,-1,-1):
    MLvspval = ax2a.plot(pvals, massloss1[i,]*days/gasses[i].dens, color=colors[i], linewidth=2.0, label=gasses[i].name)
    #MLvspval2= ax2a.plot(pvals, massloss2[3,]*days/N2.dens, color='black', linewidth=2.0, linestyle='--')

ax2a.legend(bbox_to_anchor=(1.01,1), loc='upper left', frameon=False)

fig2a.savefig('RLvsalbedo.pdf')

plt.close(fig2a)

#set up pyplot figure
fig3=plt.figure(figsize=(8,6), dpi=300)
ax3=fig3.add_subplot(111)
#fig3.subplots_adjust(left=0.1, bottom=0.1, right=0.84, top=0.9)

# Set the axis labels
ax3.set_ylabel('$a_{pred}/a_{obs}$')
ax3.set_xlabel('albedo')

# Set the axis limits
ax3.set_xlim(0.0,1.0)
ax3.set_ylim(0.0, 2.0)

# Make the tick thickness match the axes
ax3.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax3.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax3.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))
ax3.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))

pcross=np.zeros(9)

for i in np.arange(8,-1,-1):
    frac1 = NonGravAcc(gasses[i], pvals, Tsurf[i,])/ObsNonGrav(d)
    gf = np.where(frac1 > 1.0)
    if gf[0].size != 0:
        goodfracs = gf[0]
        pcross[i] = np.max(pvals[goodfracs])
    if (i == 3):
        fracvspval = ax3.plot(pvals, frac1, color='black', linewidth=2.0, label=gasses[i].name)
    else:
        fracvspval = ax3.plot(pvals, frac1, color='grey', linewidth=2.0, label=gasses[i].name)
    #fracvspval2 = ax3.plot(pvals, frac2, color='black', linewidth=2.0, linestyle='--')

frac_N2low = NonGravAcc(N2_low, pvals, Tsurf_N2low)/ObsNonGrav(d)
frac_N2high= NonGravAcc(N2_high, pvals, Tsurf_N2high)/ObsNonGrav(d)

gf = np.where(frac_N2low > 1.0)
goodfracs = gf[0]
pcross_low = np.max(pvals[goodfracs])

gf = np.where(frac_N2high > 1.0)
goodfracs = gf[0]
pcross_high = np.max(pvals[goodfracs])

pcross_err = abs(pcross_high-pcross_low)/2.0

#fvpn2l = ax3.plot(pvals, frac_N2low, color='black', linewidth=1.0, linestyle='--')
#fvpn2h = ax3.plot(pvals, frac_N2high, color='black', linewidth=1.0, linestyle='--')

#ax3.fill_between(pvals, frac_N2high, frac_N2low, color='grey', alpha=0.3)

print 'For N2, albedo at a_pred/a_obs = 1 ', pcross[3], ' +/- ', pcross_err
    
xvals = [0.1,0.2,0.4,0.4,0.4,0.3,0.2,0.4,0.96]
labelLines(plt.gca().get_lines(),xvals=xvals)

#ax3.legend(bbox_to_anchor=(1.01,1), loc='upper left', frameon=False)

#Pluto bond albedo from Buratti et al. 2017
plutobond = ax3.plot([0.72,0.72], [0.0,3.0], color='orange', linestyle='--', linewidth=2.0)
ax3.fill_between([0.65,0.79],0.0, 3.0, color='orange', alpha=0.3)
#Pluto geometric albedo (V-band) from Buratti et al. 2015 - broad band to cover both with and without opposition surge
plutogeo  = ax3.plot([0.62,0.62], [0.0,3.0], color='red', linestyle='--', linewidth=2.0)
ax3.fill_between([0.55,0.65],0.0, 3.0, color='red', alpha=0.3)

#Triton bond albedo from Hillier et al. 1990
#tritonbond = ax3.plot([0.82,0.82], [0.0,3.0], color='green', linestyle='--', linewidth=2.0)
#ax3.fill_between([0.77,0.87],0.0, 3.0, color='green', alpha=0.3)
#range of geometric albedos (V-band) from Hicks & Buratti 2004
#ax3.fill_between([0.66,0.87],0.0, 3.0, color='green', alpha=0.1)

#Size constraint from Spitzer
ax3.fill_between([0.0,0.04], 0.0, 3.0, color='purple', alpha=0.3)
ax3.fill_between([0.0,0.002], 0.0, 3.0, color='purple', alpha=0.8)
    
secax3 = ax3.twiny()
secax3.set_xlim(0.0, 1.0)
secax3.set_xticks([0.039,0.108,0.243,0.43,0.621,0.97])
secax3.set_xticklabels(['100', '60', '40', '30', '25', '20'])
secax3.set_xlabel('mean spherical radius (m)')

maxfrac = ax3.plot([0,1], [1,1], color='black', linestyle='--', linewidth=2.0)
#ax2.legend(bbox_to_anchor=(1.01,1), loc='upper left', frameon=False)

fig3.savefig('fracvsalbedo3.pdf')
plt.close(fig3)


#################################
#Orbit calculations
#
r0=1.42*aum #Distance from Sun at 12:00 on Oct 27 2017 UTC
p=pcross[3] #Use albedo that fits at point of observation
a_ell0 = La/np.sqrt(p) #a-axis of ellipsoid at time of observation
b_ell0 = Lb/np.sqrt(p) #b-axis of ellipsoid at time of observation
c_ell0 = Lc/np.sqrt(p) #c-axis of ellipsoid at time of observation
mass0  = MassTri(La, Lb, Lc, N2.dens, p)
tol = 1e-4 #target fractional change in c-axis per time step

res = optimize.root(Temp_func, [50.0], args=(N2, p, r0), method='hybr')
Tsurf0 = res.x[0]

H0 = np.arccosh((1.0-r0/a_orb)/e_orb)

t0 = np.sqrt(-a_orb**3.0/(grav*Msun))*(e_orb*np.sinh(H0)+H0)

M0 = HtoM(H0, e_orb)

print 'At ', r0/aum, ' au, outbound'
print 'Surface temperature =', Tsurf0, ' K'
print 'Mass =', mass0, ' kg'
print 'Axes, a=', a_ell0, 'm, b=', b_ell0, 'm, c=', c_ell0, 'm, ratio=', a_ell0/c_ell0

tstep0 = 86400.0 #initial timestep = 1 day
#######
#Integrate forwards

tstep = tstep0
time = t0
a_ell_t = a_ell0
b_ell_t = b_ell0
c_ell_t = c_ell0 
mass=mass0
r_orb = r0
Manom = M0

MLrate0 = MassLossRate(N2, Tsurf0)
RLrate0 = MLrate0/N2.dens

time_arrf    = np.zeros(1)
time_arrf[0] = t0

a_ell_arrf    = np.zeros(1)
b_ell_arrf    = np.zeros(1)
c_ell_arrf    = np.zeros(1)
a_ell_arrf[0] = a_ell0
b_ell_arrf[0] = b_ell0
c_ell_arrf[0] = c_ell0

r_arrf   = np.zeros(1)
r_arrf[0]= r0

mass_arrf    = np.zeros(1)
mass_arrf[0] = mass0

Tsurf_arrf    = np.zeros(1)
Tsurf_arrf[0] = Tsurf0

MLrate_arrf    = np.zeros(1)
MLrate_arrf[0] = MLrate0
RLrate_arrf    = np.zeros(1)
RLrate_arrf[0] = RLrate0



while (r_orb < 200.0*aum): #integrate out to 100 au

    RLfrac = 1.0
    
    if(r_orb < 50.0*aum): #inside 50 au we use variable timestep
        while ((RLfrac > 1.1*tol) or (RLfrac < 0.9*tol)):
            #computing radius loss rate and surface temperature at middle of time step
            #iterate until radius loss rate is 
            ntime = time + tstep/2.0
            nManom = M0 + (ntime-t0)/np.sqrt(-a_orb**3.0/(grav*Msun))
            nHanom = HyperKep(nManom, e_orb)
        
            nr_orb = a_orb*(1.0 - e_orb*np.cosh(nHanom))
        
            res = optimize.root(Temp_func, [50.0], args=(N2, p, nr_orb), method='hybr')
            Tsurf = res.x[0]
        
            MLrate = MassLossRate(N2, Tsurf)
            RL = MLrate*tstep/N2.dens
        
            RLfrac = RL/c_ell_t
        
            tstep = tstep/(RLfrac/tol)
    else: #outside 50 au, timestep stops increasing
        ntime = time + tstep/2.0
        nManom = M0 + (ntime-t0)/np.sqrt(-a_orb**3.0/(grav*Msun))
        nHanom = HyperKep(nManom, e_orb)
    
        nr_orb = a_orb*(1.0 - e_orb*np.cosh(nHanom))
    
        res = optimize.root(Temp_func, [50.0], args=(N2, p, nr_orb), method='hybr')
        Tsurf = res.x[0]
    
        MLrate = MassLossRate(N2, Tsurf)
        RL = MLrate*tstep/N2.dens
            
    #computing new radial position at end of timestep
    time = time + tstep
    time_arrf = np.append(time_arrf, time)
    
    Tsurf_arrf = np.append(Tsurf_arrf, Tsurf)
    
    Manom = M0 + (time-t0)/np.sqrt(-a_orb**3.0/(grav*Msun))
    Hanom = HyperKep(Manom, e_orb)
    
    r_orb = a_orb*(1.0 - e_orb*np.cosh(Hanom))
    r_arrf = np.append(r_arrf, r_orb)
    
    a_ell_t = a_ell_t - RL
    b_ell_t = b_ell_t - RL
    c_ell_t = c_ell_t - RL
    a_ell_arrf = np.append(a_ell_arrf, a_ell_t)
    b_ell_arrf = np.append(b_ell_arrf, b_ell_t)
    c_ell_arrf = np.append(c_ell_arrf, c_ell_t)
    
    mass = (4.0/3.0)*np.pi* a_ell_t* b_ell_t * c_ell_t *N2.dens
    mass_arrf = np.append(mass_arrf, mass)
    
    MLrate_arrf = np.append(MLrate_arrf, MLrate)
    RLrate_arrf = np.append(RLrate_arrf, MLrate/N2.dens)

#End forward integration

#print r_arrf.size
#print r_arrf[-2]/aum, r_arrf[-1]/aum
#print RLrate_arrf[-1]*1e9*years

#############

#Integrate backwards

tstep = tstep0
time = t0
a_ell_t = a_ell0
b_ell_t = b_ell0
c_ell_t = c_ell0 
mass=mass0
r_orb = r0
Manom = M0

time_arrb    = np.zeros(1)
time_arrb[0] = t0

a_ell_arrb    = np.zeros(1)
b_ell_arrb    = np.zeros(1)
c_ell_arrb    = np.zeros(1)
a_ell_arrb[0] = a_ell0
b_ell_arrb[0] = b_ell0
c_ell_arrb[0] = c_ell0

r_arrb   = np.zeros(1)
r_arrb[0]= r0

mass_arrb    = np.zeros(1)
mass_arrb[0] = mass0

Tsurf_arrb    = np.zeros(1)
Tsurf_arrb[0] = Tsurf0

MLrate_arrb    = np.zeros(1)
MLrate_arrb[0] = MLrate0
RLrate_arrb    = np.zeros(1)
RLrate_arrb[0] = RLrate0



while (r_orb < 200.0*aum): #integrate out to 100 au

    RLfrac = 1.0
    
    if(r_orb < 50.0*aum): #inside 50 au we use variable timestep
        while ((RLfrac > 1.1*tol) or (RLfrac < 0.9*tol)):
            #computing radius loss rate and surface temperature at middle of time step
            #iterate until radius loss rate is 
            ntime = time - tstep/2.0
            nManom = M0 + (ntime-t0)/np.sqrt(-a_orb**3.0/(grav*Msun))
            nHanom = HyperKep(nManom, e_orb)
        
            nr_orb = a_orb*(1.0 - e_orb*np.cosh(nHanom))
        
            res = optimize.root(Temp_func, [50.0], args=(N2, p, nr_orb), method='hybr')
            Tsurf = res.x[0]
        
            MLrate = MassLossRate(N2, Tsurf)
            RL = MLrate*tstep/N2.dens
        
            RLfrac = RL/c_ell_t
        
            tstep = tstep/(RLfrac/tol)
    else: #outside 50 au, timestep stops increasing
        ntime = time - tstep/2.0
        nManom = M0 + (ntime-t0)/np.sqrt(-a_orb**3.0/(grav*Msun))
        nHanom = HyperKep(nManom, e_orb)
    
        nr_orb = a_orb*(1.0 - e_orb*np.cosh(nHanom))
    
        res = optimize.root(Temp_func, [50.0], args=(N2, p, nr_orb), method='hybr')
        Tsurf = res.x[0]
    
        MLrate = MassLossRate(N2, Tsurf)
        RL = MLrate*tstep/N2.dens
            
    #computing new radial position at end of timestep
    time = time - tstep
    time_arrb = np.append(time_arrb, time)
    
    Tsurf_arrb = np.append(Tsurf_arrb, Tsurf)
    
    Manom = M0 + (time-t0)/np.sqrt(-a_orb**3.0/(grav*Msun))
    Hanom = HyperKep(Manom, e_orb)
    
    r_orb = a_orb*(1.0 - e_orb*np.cosh(Hanom))
    r_arrb = np.append(r_arrb, r_orb)
    
    a_ell_t = a_ell_t + RL
    b_ell_t = b_ell_t + RL
    c_ell_t = c_ell_t + RL
    a_ell_arrb = np.append(a_ell_arrb, a_ell_t)
    b_ell_arrb = np.append(b_ell_arrb, b_ell_t)
    c_ell_arrb = np.append(c_ell_arrb, c_ell_t)
    
    mass = (4.0/3.0)*np.pi* a_ell_t* b_ell_t * c_ell_t *N2.dens
    mass_arrb = np.append(mass_arrb, mass)
    
    MLrate_arrb = np.append(MLrate_arrb, MLrate)
    RLrate_arrb = np.append(RLrate_arrb, MLrate/N2.dens)  
    
# End backward integration
###############

# Output a few orbital locations to file

datfile = open('erosionintegration.dat', 'w')

datfile.write('    d (au)    time (days)    Tsurf (K)     2a (m)       2b (m)       2c (m)        a/c       mass (kg)    Rad loss rate (m/day)\n')

datfile.write('{0:10.6E} {1:10.6E} {2:10.6E} {3:10.6E} {4:10.6E} {5:10.6E} {6:10.6E} {7:10.6E} {8:10.6E}\n'.format(r_arrb[-1]/aum, (time_arrb[-1]-t0)/days, Tsurf_arrb[-1], 2.0*a_ell_arrb[-1], 2.0*b_ell_arrb[-1], 2.0*c_ell_arrb[-1], a_ell_arrb[-1]/c_ell_arrb[-1], mass_arrb[-1], RLrate_arrb[-1]*days))

index1 = np.min(np.where(r_arrb/aum > 130.0))
index2 = np.min(np.where(r_arrb/aum > 100.0))
index3 = np.min(np.where(r_arrb/aum > 50.0))
index4 = np.min(np.where(r_arrb/aum > 30.0))
index5 = np.min(np.where(r_arrb/aum > 19.0))
index6 = np.min(np.where(r_arrb/aum > 9.6))
index7 = np.min(np.where(r_arrb/aum > 5.2))
index8 = np.min(np.where(r_arrb/aum > 2.0))
index9 = np.max(np.where(r_arrb/aum < 1.0))
index10 = np.max(np.where(r_arrb/aum <= np.min(r_arrb/aum)))
index11 = np.min(np.where(r_arrb/aum < 1.0))

indices = [index1, index2, index3, index4, index5, index6, index7, index8, index9, index10, index11]

for ind in indices:
    datfile.write('{0:10.6E} {1:10.6E} {2:10.6E} {3:10.6E} {4:10.6E} {5:10.6E} {6:10.6E} {7:10.6E} {8:10.6E}\n'.format(r_arrb[ind]/aum, (time_arrb[ind]-t0)/days, Tsurf_arrb[ind], 2.0*a_ell_arrb[ind], 2.0*b_ell_arrb[ind], 2.0*c_ell_arrb[ind], a_ell_arrb[ind]/c_ell_arrb[ind], mass_arrb[ind], RLrate_arrb[ind]*days))

datfile.write('{0:10.6E} {1:10.6E} {2:10.6E} {3:10.6E} {4:10.6E} {5:10.6E} {6:10.6E} {7:10.6E} {8:10.6E}\n'.format(r0/aum, 0.0, Tsurf0, 2.0*a_ell0, 2.0*b_ell0, 2.0*c_ell0, a_ell0/c_ell0, mass0, RLrate0*days))

index1 = np.min(np.where(r_arrf/aum > 2.0))
index2 = np.min(np.where(r_arrf/aum > 5.2))
index3 = np.min(np.where(r_arrf/aum > 9.6))
index4 = np.min(np.where(r_arrf/aum > 19.0))
index5 = np.min(np.where(r_arrf/aum > 30.0))
index6 = np.min(np.where(r_arrf/aum > 50.0))
index7 = np.min(np.where(r_arrf/aum > 100.0))
index8 = np.min(np.where(r_arrf/aum > 130.0))

indices = [index1, index2, index3, index4, index5, index6, index7, index8]

for ind in indices:
    datfile.write('{0:10.6E} {1:10.6E} {2:10.6E} {3:10.6E} {4:10.6E} {5:10.6E} {6:10.6E} {7:10.6E} {8:10.6E}\n'.format(r_arrf[ind]/aum, (time_arrf[ind]-t0)/days, Tsurf_arrf[ind], 2.0*a_ell_arrf[ind], 2.0*b_ell_arrf[ind], 2.0*c_ell_arrf[ind], a_ell_arrf[ind]/c_ell_arrf[ind], mass_arrf[ind], RLrate_arrf[ind]*days))
    
datfile.write('{0:10.6E} {1:10.6E} {2:10.6E} {3:10.6E} {4:10.6E} {5:10.6E} {6:10.6E} {7:10.6E} {8:10.6E}\n'.format(r_arrf[-1]/aum, (time_arrf[-1]-t0)/days, Tsurf_arrf[-1], 2.0*a_ell_arrf[-1], 2.0*b_ell_arrf[-1], 2.0*c_ell_arrf[-1], a_ell_arrf[-1]/c_ell_arrf[-1], mass_arrf[-1], RLrate_arrf[-1]*days))

datfile.close()



time_arrb  = np.flip(time_arrb, 0)
times = np.append(time_arrb, t0)
times = np.concatenate((times, time_arrf))

Tsurf_arrb = np.flip(Tsurf_arrb, 0)
Tsurfs = np.append(Tsurf_arrb, Tsurf0)
Tsurfs = np.concatenate((Tsurfs, Tsurf_arrf))

r_arrb     = np.flip(r_arrb, 0)
rads = np.append(r_arrb, r0)
rads = np.concatenate((rads, r_arrf))

a_ell_arrb = np.flip(a_ell_arrb, 0)
a_ells = np.append(a_ell_arrb, a_ell0)
a_ells = np.concatenate((a_ells, a_ell_arrf))

b_ell_arrb = np.flip(b_ell_arrb, 0)
b_ells = np.append(b_ell_arrb, b_ell0)
b_ells = np.concatenate((b_ells, b_ell_arrf))

c_ell_arrb = np.flip(c_ell_arrb, 0)
c_ells = np.append(c_ell_arrb, c_ell0)
c_ells = np.concatenate((c_ells, c_ell_arrf))

mass_arrb  = np.flip(mass_arrb, 0)
masses = np.append(mass_arrb, mass0)
masses = np.concatenate((masses, mass_arrf))

MLrate_arrb  = np.flip(MLrate_arrb, 0)
MLrates = np.append(MLrate_arrb, MLrate0)
MLrates = np.concatenate((MLrates, MLrate_arrf))

RLrate_arrb  = np.flip(RLrate_arrb, 0)
RLrates = np.append(RLrate_arrb, RLrate0)
RLrates = np.concatenate((RLrates, RLrate_arrf))

#print r_arrb.size
#print r_arrb[-2]/aum, r_arrb[-1]/aum
#print RLrate_arrb[-1]*1e9*years

res = np.where(rads == np.min(rads))
peri = res[0][0]

print 'At perihelion, ', np.min(rads)/aum, 'au'
print 'Surface temperature =', np.max(Tsurfs), 'K'
print 'Radius loss rate =', np.max(RLrates)*days, 'm/day'
print 'Mass =', masses[peri], 'kg'
print 'Axes, a=', a_ells[peri], 'm, b=', b_ells[peri], 'm, c=', c_ells[peri], 'm, ratio=', a_ells[peri]/c_ells[peri]
print 'At ', rads[-1]/aum, ' au, outbound'
print 'Mass =', masses[-1], ' kg'
print 'Axes, a=', a_ells[-1], 'm, b=', b_ells[-1], 'm, c=', c_ells[-1], 'm, ratio=', a_ells[-1]/c_ells[-1]
print 'At ', rads[0]/aum, ' au, inbound'
print 'Mass =', masses[0], ' kg'
print 'Axes, a=', a_ells[0], 'm, b=', b_ells[0], 'm, c=', c_ells[0], 'm, ratio=', a_ells[0]/c_ells[0]

#set up pyplot figure
fig4=plt.figure(figsize=(12,5), dpi=300)
ax4a=fig4.add_subplot(221)
#fig4.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

# Set the axis labels
ax4a.set_ylabel('dist. from Sun (au)')
#ax4a.set_xlabel('time (days)')

# Set the axis limits
ax4a.set_xlim(-550.0,500.0)
ax4a.set_ylim(0.0, 12.0)

# Make the tick thickness match the axes
ax4a.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax4a.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax4a.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(50.0))
ax4a.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(1.0))

radplot = ax4a.plot((times-t0)/days, rads/aum, color='black', linewidth=2.0)

ax4b=fig4.add_subplot(222, sharex=ax4a)

# Set the axis labels
ax4b.set_ylabel('mass (x10$^6$ kg)')

# Set the axis limits
ax4b.set_xlim(-550.0,500.0)
ax4b.set_ylim(0.0, 100.0)

# Make the tick thickness match the axes
ax4b.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax4b.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax4b.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(50.0))
ax4b.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(5.0))

massplot = ax4b.plot((times-t0)/days, masses/1e6, color='black', linewidth=2.0)

ax4c=fig4.add_subplot(224, sharex=ax4a)

# Set the axis labels
ax4c.set_ylabel('axis ratio')
ax4c.set_xlabel('time (days)')

# Set the axis limits
ax4c.set_xlim(-550.0,500.0)
ax4c.set_ylim(0.0, 10.0)

# Make the tick thickness match the axes
ax4c.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax4c.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax4c.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(50.0))
ax4c.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.5))

ratplot = ax4c.plot((times-t0)/days, a_ells/c_ells, color='black', linewidth=2.0)

ax4d=fig4.add_subplot(223, sharex=ax4a)

# Set the axis labels
ax4d.set_ylabel('surf. temp. (K)')
ax4d.set_xlabel('time (days)')

# Set the axis limits
ax4d.set_xlim(-550.0,500.0)
ax4d.set_ylim(30.0, 50.0)

# Make the tick thickness match the axes
ax4d.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax4d.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax4d.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(50.0))
ax4d.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.5))

Tempsplot = ax4d.plot((times-t0)/days, Tsurfs, color='black', linewidth=2.0)

fig4.savefig('timeplot.pdf')
plt.close(fig4)

#

gravacc = grav * Msun / (rads**2.0)

La = a_ells*np.sqrt(p)
Lb = b_ells*np.sqrt(p)
Lc = c_ells*np.sqrt(p)

nongravacc = NonGravAcc(N2, p, Tsurfs)

ngar2 = nongravacc * (rads/aum)**2.0

#nga2r2 = 4.55e-6 * (rads/aum)**0.22
#nga2 = 4.55e-6 * (aum/rads)**1.78
nga2 = 4.59e-6 * (aum/rads)**1.8
nga2r2= nga2 * (rads/aum)**2.0

obga = 4.92e-6 * (aum/rads)**2.0
obgar2= 4.92e-6*np.ones(times.size)
ob2ga = 3.46e-6 * (aum/rads)
ob2gar2= ob2ga * (rads/aum)**2.0

print np.max(nongravacc)

#set up pyplot figure
fig5a=plt.figure(figsize=(12,6), dpi=300)
ax5a=fig5a.add_subplot(121)
#fig4.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

# Set the axis labels
ax5a.set_ylabel('$A_{non grav}$ x10$^{-5}$ m s$^{-2}$')
ax5a.set_xlabel('time (days)')

# Set the axis limits
ax5a.set_xlim(-50.0,100.0)
ax5a.set_ylim(0.0, 1.0)

# Make the tick thickness match the axes
ax5a.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax5a.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax5a.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(5.0))
ax5a.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))

betaplota= ax5a.plot((times-t0)/days, nga2*1e5, color='black', linewidth=2.0)
betaplot = ax5a.plot((times-t0)/days, nongravacc*1e5, color='red', linewidth=2.0)
betaplot2= ax5a.plot((times-t0)/days, obga*1e5, color='black', linewidth=2.0, linestyle='--')
betaplot3= ax5a.plot((times-t0)/days, ob2ga*1e5, color='black', linewidth=2.0, linestyle='dotted')
#betaplot2u= ax5.plot((times-t0)/days, 5.08e-6*np.ones(times.size), color='black', linewidth=1.0, linestyle='--')
#betaplot2d= ax5.plot((times-t0)/days, 4.76e-6*np.ones(times.size), color='black', linewidth=1.0, linestyle='--')

#fig5a.savefig('betaplot2.pdf')
#plt.close(fig5a)

#set up pyplot figure
#fig5=plt.figure(figsize=(8,6), dpi=300)
ax5=fig5a.add_subplot(122)
#fig4.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

# Set the axis labels
ax5.set_ylabel('$A_{non grav}$ x10$^{-5}$ (r/au)$^2$ m s$^{-2}$')
ax5.set_xlabel('time (days)')

# Set the axis limits
ax5.set_xlim(-50.0,100.0)
ax5.set_ylim(0.0, 1.0)

# Make the tick thickness match the axes
ax5.tick_params(width=2.0, length=10.0, direction='in', top='on', right='on')
ax5.tick_params(which='minor', width=1.0, length=5.0, direction='in', bottom='on', left='on', top='on', right='on')

ax5.xaxis.set_minor_locator(mpt.ticker.MultipleLocator(5.0))
ax5.yaxis.set_minor_locator(mpt.ticker.MultipleLocator(0.05))

betaplota= ax5.plot((times-t0)/days, nga2r2*1e5, color='black', linewidth=2.0)
betaplot = ax5.plot((times-t0)/days, ngar2*1e5, color='red', linewidth=2.0)
betaplot2= ax5.plot((times-t0)/days, obgar2*1e5, color='black', linewidth=2.0, linestyle='--')
betaplot3= ax5.plot((times-t0)/days, ob2gar2*1e5, color='black', linewidth=2.0, linestyle='dotted')
#betaplot2u= ax5.plot((times-t0)/days, 5.08e-6*np.ones(times.size), color='black', linewidth=1.0, linestyle='--')
#betaplot2d= ax5.plot((times-t0)/days, 4.76e-6*np.ones(times.size), color='black', linewidth=1.0, linestyle='--')

fig5a.savefig('betaplot3.pdf')
plt.close(fig5a)