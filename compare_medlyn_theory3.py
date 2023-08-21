# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:10:07 2022

@author: natan
"""


import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
# import pandas as pd
import statsmodels.api as sm
import scipy.optimize
# import glob
# import statsmodels.formula.api as smf

# import matplotlib as mpl
#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8

plt.rcParams['font.size']=26
plt.rcParams["mathtext.default"] = "sf"
#%%
g_range = np.linspace(-0.2,2,600)
vmax = 15
slope_at_0 = 110
a_exp = vmax*(1-np.exp(-g_range/vmax*slope_at_0))
#%%
vpd_range = np.linspace(0.5,3,500)
tau_day = 30
tau_s = tau_day*(60*60*24)
zsoil_mol = 1000*1000/18
zsoil_mm = 1000
s_above_wilt = 0.2
new_opt_g = np.sqrt(2/tau_s*vmax/slope_at_0*zsoil_mol*s_above_wilt/(vpd_range/100))
#%%
g1 = 5.33
quot_pred = 1.6/400*(1+g1/np.sqrt(vpd_range))
med_g = np.interp(quot_pred,g_range/a_exp,g_range)
med_g[med_g < 0] = 0
#%%
fig,axes = plt.subplots(3,2,figsize=(15,18))
ax = axes[0,0]
#plt.figure(figsize=(10,6))
ax.plot(vpd_range,new_opt_g,"r",label="Time-integrated",linewidth=3); 
ax.plot(vpd_range,med_g,"k--",label="Medlyn",linewidth=3)
ax.set_xlabel("VPD (kPa)")
ax.set_ylabel("g $(mol/m^2/s)$")
ax.legend(fontsize=24)
ax.set_title("(a)",loc="left")
#%%
vpd_const = 1
tau_day = 30
tau_s = tau_day*(60*60*24)
zsoil_mol = 1000*1000/18
s_sw_range = np.linspace(0.01,0.3,500)
new_opt_g = np.sqrt(2/tau_s*vmax/slope_at_0*zsoil_mol*s_sw_range/(vpd_const/100))
#%%
quot_pred = 1.6/400*(1+g1*(s_sw_range/0.2)**1/np.sqrt(vpd_const))
med_g = np.interp(quot_pred,g_range/a_exp,g_range)
med_g[med_g < 0] = 0

#%%

# plt.figure()
# plt.plot(s_sw_range,new_opt_g,"r",label="Time-integrated",linewidth=3); 
# plt.plot(s_sw_range,med_g,"k--",label="Medlyn",linewidth=3)
# plt.xlabel("Soil moisture above wilting point")
# plt.ylabel("g $(mol/m^2/s)$")
# plt.legend()
#%%
quot_soil = new_opt_g/(vmax*(1-np.exp(-new_opt_g/vmax*slope_at_0)))
g1_soil = (quot_soil*400/1.6 - 1)*np.sqrt(vpd_const)/g1
#%%
mystack = np.stack((s_sw_range,np.sqrt(s_sw_range)),1)
beta_mod = sm.OLS(g1_soil,  sm.add_constant(mystack),missing='drop').fit()
#%%
def toopt(pars):
    q,offset,coef = pars
    betapred = coef*np.clip(s_sw_range + offset,0,np.inf)**q
    return (betapred - g1_soil)[np.isfinite(g1_soil)]
fit0 = np.array([0.5,0.01,2])
myopt = scipy.optimize.least_squares(toopt,x0=fit0,method="lm",x_scale=np.abs(fit0))
#%%

# plt.figure()
# plt.plot(s_sw_range,g1_soil,"k",label="function that matches new model",linewidth=3)
# plt.plot(s_sw_range,beta_mod.predict(sm.add_constant(mystack)),":",color="blue",label="linear+sqrt fit",linewidth=3)
# plt.plot(s_sw_range,myopt.x[2]*(s_sw_range+myopt.x[1])**myopt.x[0],"--",color="orange",label="Shifted power law",linewidth=3)

# plt.xlabel("Soil moisture above wilting point")
# plt.ylabel("beta function value")
# plt.ylim(0,1.3)

# plt.legend(loc="lower right")
#%%
#beta_value = beta_mod.predict(sm.add_constant(mystack))
beta_value = myopt.x[2]*(s_sw_range+myopt.x[1])**myopt.x[0]
quot_pred = 1.6/400*(1+g1*beta_value/np.sqrt(vpd_const))
med_g = np.interp(quot_pred,g_range/a_exp,g_range)
med_g[med_g < 0] = 0

#%%
ax = axes[0,1]
ax.plot(s_sw_range*zsoil_mm,new_opt_g,"r",label="Time-integrated",linewidth=3); 
ax.plot(s_sw_range*zsoil_mm,med_g,"k--",label=r"Medlyn",linewidth=3)
ax.set_xlabel("s - $s_w$ (mm)")
ax.set_ylabel("g $(mol/m^2/s)$")
ax.set_title("(b)",loc="left")

#ax.legend(loc="lower right",fontsize=18)
# plt.figure(figsize=(10,6))
# plt.plot(s_sw_range,new_opt_g,"r",label="Time-integrated",linewidth=3); 
# plt.plot(s_sw_range,med_g,"k--",label=r"Medlyn with calibrated $\beta(s)$",linewidth=3)
# plt.xlabel("Soil moisture above wilting point")
# plt.ylabel("g $(mol/m^2/s)$")
# plt.legend(loc="lower right")
#%%
vpd_range = np.linspace(0.5,3,20).reshape(1,-1)
tau_day = 30
tau_s = tau_day*(60*60*24)
zsoil_mol = 1000*1000/18
s_sw_range = np.linspace(0.01,0.3,20).reshape(-1,1)
new_opt_g = np.sqrt(2/tau_s*vmax/slope_at_0*zsoil_mol*s_sw_range/(vpd_range/100))
#%%
#beta_value = beta_mod.params[0] + beta_mod.params[1]*s_sw_range + beta_mod.params[2]*np.sqrt(s_sw_range)
beta_value = myopt.x[2]*(s_sw_range+myopt.x[1])**myopt.x[0]

quot_pred = 1.6/400*(1+g1*beta_value/np.sqrt(vpd_range))
med_g = np.interp(quot_pred,g_range/a_exp,g_range)
med_g[med_g < 0] = 0
#%%
allmax = np.round(max(np.max(new_opt_g),np.max(med_g)),1)
#%%



ax = axes[1,0]
pcm = ax.pcolormesh(vpd_range,s_sw_range*zsoil_mm,new_opt_g,vmin=0,vmax=allmax)
ax.set_xlabel("VPD (kPa)")
ax.set_ylabel("s - $s_w$ (mm)")
#ax.set_title("$g_{INT}$ $(mol/m^2/s)$")
ax.set_title("(c)",loc="left")

fig.colorbar(pcm, ax=ax,label="$g_{INT}$ $(mol/m^2/s)$")


ax = axes[1,1]
pcm = ax.pcolormesh(vpd_range,s_sw_range*zsoil_mm,med_g,vmin=0,vmax=allmax)
ax.set_xlabel("VPD (kPa)")
ax.set_ylabel("s - $s_w$ (mm)")
ax.set_title("(d)",loc="left")

fig.colorbar(pcm, ax=ax,label="$g_{MED}$ $(mol/m^2/s)$")


#ax.colorbar()
gdiff = new_opt_g-med_g
absmax = np.max(np.abs(gdiff))
ax = axes[2,0]
pcm = ax.pcolormesh(vpd_range,s_sw_range*zsoil_mm,gdiff,cmap="RdBu",vmin=-absmax,vmax= absmax)
ax.set_xlabel("VPD (kPa)")
ax.set_ylabel("s - $s_w$ (mm)")
#ax.set_title("$g_{INT}-g_{MED}$ $(mol/m^2/s)$")
ax.set_title("(e)",loc="left")

fig.colorbar(pcm, ax=ax,label="$g_{INT}-g_{MED}$ $(mol/m^2/s)$")
#ax.colorbar()

ax = axes[2,1]
ax.plot(new_opt_g.reshape(-1,1),med_g.reshape(-1,1),'.')
ax.plot([0,0.6],[0,0.6],linewidth=3)
ax.set_xlabel("$g_{INT}$ $(mol/m^2/s)$")
ax.set_ylabel("$g_{MED}$ $(mol/m^2/s)$")
ax.set_title("(f)",loc="left")


plt.tight_layout()
