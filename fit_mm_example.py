# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:54:18 2023

@author: nholtzma
"""

import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import statsmodels.api as sm
import scipy.optimize
import glob
import statsmodels.formula.api as smf

import matplotlib as mpl
#%%
bif_data = pd.read_csv("../optimality_proj/fn2015_bif_tab.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])]
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH"])]
metadata = pd.read_csv(r"C:\Users\nholtzma\Downloads\fluxnet_site_info_all.csv")

all_daily = glob.glob(r"C:\Users\nholtzma\Downloads\fluxnet2015\daily_data\*.csv")
forest_daily = [x for x in all_daily if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
#%%

all_hh = glob.glob(r"C:\Users\nholtzma\Downloads\fluxnet2015\*_HH_*.csv")
forest_hh = [x for x in all_hh if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
all_hourly = glob.glob(r"C:\Users\nholtzma\Downloads\fluxnet2015\*_HR_*.csv")
forest_h = [x for x in all_hourly if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
forest_all = forest_hh + forest_h
#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%s
#latsort = bif_forest.sort_values("LOCATION_LAT")
#%%
#test_sites = np.array(latsort.SITE_ID)[::5]
#%%
all_daily = pd.read_csv("dailydata_aug21.csv",parse_dates=["date"])
#%%
#all_evi = pd.read_csv("flx_all_lai2.csv")
#%%
site_tab = []
#%%
for site in ["US-MMS"]:
    print(site)
    try:
        fname = [x for x in forest_all if site in x][0]
    except:
        continue
    #%%
    # df = pd.read_csv(fname,parse_dates=["TIMESTAMP_START"])
    
    # df["date"] = df["TIMESTAMP_START"].dt.date
    #%%
    df = pd.read_csv(fname)
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"],format="%Y%m%d%H%M")
    df["date"] = df["TIMESTAMP_START"].dt.date
    df["hour"] = df.TIMESTAMP_START.dt.hour
    df["doy_raw"] = df.TIMESTAMP_START.dt.dayofyear
    df["year_raw"] = df.TIMESTAMP_START.dt.year
    df = df.loc[df.year_raw >= 2001].copy()
    #%%
    if len(df) < 25*24:
        continue
#df["year"] = pd.to_datetime(df.TIMESTAMP_START).dt.year
    #%%
    dfull = all_daily.loc[all_daily.SITE_ID==site].copy()
    
    dcount = dfull.groupby("year_new").count().reset_index()
    fullyear = dcount.year_new.loc[dcount.date > 300]
    dfull = dfull.loc[dfull.year_new.isin(fullyear)]
    #%%
    if np.max(dfull.LAI) < 0.05:
        continue
    if len(dfull) < 300:
        continue
    #%%
    if np.max(dfull.LAI) < 0.05:
        continue
    #%%
    # laimms = all_evi.loc[all_evi.SITE_ID==site].copy().rename(columns={"mean":"LAI2"})
    # laimms["datecode"] = laimms["system:index"].str.slice(stop=10)
    # laimms["date"] = pd.to_datetime(laimms["datecode"],format="%Y_%m_%d")#.dt.date
    # #%%
    # dfull = pd.merge(dfull,laimms[["date","LAI2"]],on="date",how="left")
    # #%%
    # if np.sum(np.isfinite(dfull.LAI2)) == 0:
    #     dfull["LAI3"] = np.nan
    
    # else:
    #     lai_arr = np.array(dfull.LAI2)
        
    #     lai_int = np.interp(np.arange(len(lai_arr)),
    #                         np.arange(len(lai_arr))[np.isfinite(lai_arr)],
    #                         lai_arr[np.isfinite(lai_arr)],
    #                         left=np.nan,right=np.nan)    
    #     dfull["LAI3"] = lai_int/10
        
    #     lai_med = np.zeros(len(lai_arr))
    #     w = 15
    #     for j in range(w,len(lai_arr)-w):
    #         lai_med[j] = np.nanmean(lai_arr[j-w:j+w+1])
    #     lai_med[:w] = lai_med[w]
    #     lai_med[-w:] = lai_med[-w]
    
    
    #%%
    #dfull["gpp"] = (dfull.gpp_dt+dfull.gpp_nt)/2
    #%%
    
    #evi_stretch = (dfull["EVI2"]-np.nanmean(dfull.EVI2))/np.nanstd(dfull.EVI2)*np.nanstd(dfull.LAI) + np.nanmean(dfull.LAI)
    
    # evi_stretch = np.interp(dfull.EVI2,np.sort(dfull.EVI2),np.sort(dfull.LAI))
    
    # dfull["LAI"] = evi_stretch
    #%%
    if np.mean(np.isfinite(dfull.LAI)) < 0.5:
        continue
#%%
    dclim = dfull.groupby("doy").mean(numeric_only=True).reset_index()
    #gpp_clim = np.array(dclim.gpp)
    year95 = dfull.groupby("year_new").max(numeric_only=True).reset_index()

    #year95["gpp_y95"] = 1*year95["gpp_smooth"]
    year95["lai_y95"] = 1*year95["LAI"]
    dfull = pd.merge(dfull,year95[["year_new","lai_y95"]],how="left",on="year_new")
    dfull["LAI_gt50"] = dfull.LAI/dfull.lai_y95 > 0.75

    #df["LAI_gt50"] = (df.gpp_smooth/df.gpp_y95) > 0.67
    year_list = pd.unique(dfull.year_new)
    gs_starts = []
    gs_ends = []
    for year in year_list:
        dfy = dfull.loc[dfull.year_new==year].reset_index()
        topday = np.argmax(dfy.LAI)
#        topday = np.argmax(dfy.gpp_smooth)

        under50 = np.where(~dfy.LAI_gt50)[0]
        try:
            summer_start = under50[under50 < topday][-1] + 1
        except:
            summer_start = 0
        try:
            summer_end = under50[under50 > topday][0] - 1
        except:
            summer_end = 365
        gs_starts.append(summer_start)
        gs_ends.append(summer_end)
        
    summer_df = pd.DataFrame({"year_new":year_list,
                              "summer_start":gs_starts,
                              "summer_end":gs_ends})
    dfull= pd.merge(dfull,summer_df,on="year_new",how="left")
    is_summer = np.array((dfull.doy >= dfull.summer_start)*(dfull.doy <= dfull.summer_end))
    
    gpp_clim_std = np.array(dclim.LAI)/np.nanmax(dclim.LAI)
    topday = np.argmax(gpp_clim_std)
    under50 = np.where(gpp_clim_std < 0.75)[0]
    try:
        clim_summer_start = under50[under50 < topday][-1] + 1
    except:
        clim_summer_start = 0
    try:
        clim_summer_end = under50[under50 > topday][0] -1
    except:
        clim_summer_end = 365
    dfull["clim_summer"] = (dfull.doy >= clim_summer_start)*(dfull.doy <= clim_summer_end)
    #is_summer *= clim_summer
    
    daily_gs = dfull.loc[is_summer].copy()
    
    daily_gs["date"] = daily_gs["date"].dt.date
    #%%
    df = pd.merge(df,daily_gs[["date","LAI","lai_y95","year_new"]],on="date",how="inner")
    #%%
    
    # daily_gs = dfull.loc[dfull.is_summer].copy()
    
    # daily_gs["date"] = daily_gs["date"].dt.date
    # #%%
    # df = pd.merge(df,daily_gs[["date","LAI","year_new"]],on="date",how="inner")
    
    
    #%%
    
    
    #%%
    
    df[df == -9999] = np.nan
    #%%
    df["PPFD_in"] = df.SW_IN_F
    df["VPD"] = df.VPD_F/10
    df["LE"] = np.clip(df["LE_F_MDS"],0,np.inf)
    g1 = np.clip(0.5*(df.GPP_NT_VUT_REF + df.GPP_DT_VUT_REF),0,np.inf)
    g1[df.GPP_NT_VUT_REF < 0] = np.nan
    g1[df.GPP_DT_VUT_REF < 0] = np.nan
    g1[df.LE == 0] = 0
    g1[df.PPFD_in == 0] = 0

    df["gpp"] = g1
    
    df["T_AIR"] = df.TA_F
    df["cond"] = df.LE/44200/(df.VPD/100)
    #%%
    
    #%%
    dfday = df.loc[df.PPFD_in > 100].copy()
    dfday = dfday.loc[dfday.VPD > 0.5].copy()
    dfday = dfday.loc[dfday.LE > 0].copy()
    dfday = dfday.loc[dfday.P_F == 0].copy()

    dfday = dfday.loc[np.isfinite(dfday.gpp)].copy()
    dfday = dfday.loc[dfday.LE_F_MDS_QC <= 1]
    
    dfday = dfday.loc[np.isfinite(dfday.gpp)]
    
    dfday = dfday.loc[dfday.gpp > 0].copy()
    
    dfday = dfday.loc[dfday.NEE_VUT_REF_QC <= 1].copy()
    #%%
    relwue = dfday.gpp/dfday.cond
    dfday = dfday.loc[relwue < 500].copy()

#%%
    if len(dfday) < 25:
        continue
#%% 
    #dfday = dfday.loc[dfday.cond >= 0.1]

#%%    
    #%%
    #laimean = np.mean(dfday.LAI_modis)
    #laifac = 1
    #sigma = 20
    #%%
    
    #%%
    dfday["cond_norm"] = dfday.cond/dfday.LAI
    dfday["gpp_norm"] = dfday.gpp/dfday.LAI
    #%%
    # yearq9 = dfday.groupby("year_new").quantile(0.9,numeric_only=True).reset_index()
    # yearq9["cond_norm_q9"] = yearq9.cond_norm
    # dfday = pd.merge(dfday,yearq9[["year_new","cond_norm_q9"]],on="year_new",how="left")
    
    # dfhi = dfday.loc[dfday.cond_norm > dfday.cond_norm_q9].copy()
    dfhi = dfday.loc[dfday.cond_norm > np.quantile(dfday.cond_norm,0.9)]
    #%%
    #dflo  = dfday.loc[dfday.PPFD_in < np.quantile(dfday.PPFD_in,0.1)]

    #%%
#    himod = smf.ols('np.log(gpp_norm) ~ np.log(PPFD_in) + T_AIR + np.power(T_AIR,2) + C(year_new)',data=dfhi,missing='drop').fit()
    himod = smf.ols('np.log(gpp_norm) ~ np.log(PPFD_in) + T_AIR + np.power(T_AIR,2)',data=dfhi,missing='drop').fit()
    dfday["amax"] = np.exp(himod.predict(dfday))
    # def tofit(pars):
    #     toptA,parkA,maxA = pars
    #     tfac = np.exp(-(dfhi.T_AIR-toptA)**2/20**2/2) / np.exp(-(25-toptA)**2/20**2/2)
    #     rfac = (dfhi.PPFD_in) / (dfhi.PPFD_in + parkA) / (800/(800+parkA))
    #     gpred = maxA*tfac*rfac
    #     return gpred-dfhi.gpp_norm
    # fit0 = np.array([25,500,np.mean(dfhi.gpp_norm)])
    # myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))


    # #%%
    # toptA,parkA,maxA = myfit.x
    # tfac = np.exp(-(dfday.T_AIR-toptA)**2/20**2/2) / np.exp(-(25-toptA)**2/20**2/2)
    # rfac = (dfday.PPFD_in) / (dfday.PPFD_in + parkA) / (800/(800+parkA))
    # gpred = maxA*tfac*rfac
    
    
    # dfday["amax"] = gpred
    # #%%
    dfday["gA_topred"] = -dfday.cond_norm/np.log(1-dfday.gpp_norm/dfday.amax)
    m30 = smf.ols("np.log(gA_topred) ~ np.log(PPFD_in) + T_AIR + np.power(T_AIR,2)",data=dfday,missing="drop").fit()
    #%%
    dfday["gA_pred"] = np.exp(m30.predict(dfday))
    # #%%
    # def tofit(pars):
    #     amax,g = pars
    #     gpp_pred = amax*dfday.amax*(1-np.exp(-dfday.cond_norm/dfday.amax/amax*g))
    #     return gpp_pred - dfday.gpp_norm
    # fit0 = np.array([1,150])
    # myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))
    # afac,slope = myfit.x
    # #%%
    # df = df.loc[df.year_new.isin(pd.unique(dfday.year_new))].copy()
    #%% 
    
    #%%
    df["amax_hourly"] = np.exp(himod.predict(df)) * df.LAI

    df["gA_hourly"] = np.exp(m30.predict(df)) * df.LAI
    df["gpp_pred_hourly"] = df["amax_hourly"] * (1 - np.exp(-df.cond/df["gA_hourly"]))
    #%%
    #df["gpp_pred"] = gpp_pred
    #df["kgpp"] = k_pred
    #%%
    # gpp2 = np.array(df.gpp)
    # for hi in range(24):
    #     z = gpp2[hi::24]*1
    #     z2 = np.interp(np.arange(len(z)), np.arange(len(z))[np.isfinite(z)], z[np.isfinite(z)])
    #     gpp2[hi::24] = z2
    
    # df["gpp_interp"] = gpp2

    #%%
    diurnal = df.groupby("hour").mean(numeric_only=True).reset_index()
    #%%
    fillgpp = 1*df.gpp
    fillgpp[np.isnan(df.gpp)] = np.array(diurnal.gpp)[np.array(df.hour[np.isnan(df.gpp)])]
    
    df["gpp_fill"] = fillgpp
    #%%
    df3 = df.loc[np.isfinite(df.gpp)].copy()
    
    #%%
    daytime_avg = df3.loc[df.NIGHT==0].groupby("date").mean(numeric_only=True).reset_index()
    #%%
    dailydf = df3.groupby("date").mean(numeric_only=True).reset_index()
    #%%
    dailydf["cond_daily_dayVPD"] = dailydf.LE/44200/(daytime_avg.VPD/100)

    #%%
    # teff = np.exp(-(daytime_avg.T_AIR-topt1)**2/2/sigma1**2) / np.exp(-(25-topt1)**2/2/sigma1**2)
    # radeff = 1-np.exp(-daytime_avg.PPFD_in/parK)
    # k_pred = gmax*teff*radeff*daytime_avg.LAI
    #%%
    dailydf["cond_daytime"] = daytime_avg.LE/44200/(daytime_avg.VPD/100)
    dailydf["vpd_daytime"] = daytime_avg.VPD
    dailydf["T_AIR"] = daytime_avg.T_AIR
    dailydf["PPFD_in"] = daytime_avg.PPFD_in

    #%%
    dailydf["dayfrac1"] = 1-dailydf["NIGHT"]
    #k_pred = daytime_avg.kgpp
    #%%
    #dailydf["gA_daily"] = np.interp(daytime_avg.PPFD_in,parmed,empK) * dailydf.LAI * dailydf.dayfrac1
    #dailydf["amax_daily"] = np.interp(daytime_avg.PPFD_in,parmed,empAmax) * dailydf.LAI * dailydf.dayfrac1
    
    # dailydf["amax_daily"] = np.exp(himod.predict(dailydf)) * dailydf.LAI * afac * dailydf.dayfrac1
    # dailydf["gA_daily"] = dailydf["amax_daily"] / slope
    
    # dailydf["gpp_pred_daily"] = dailydf["amax_daily"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_daily"]))
    # #%%
    #dailydf["gpp_pred_hourly2"] = dailydf["amax_hourly"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_hourly"]))
    dailydf["amax_daily"] = np.exp(himod.predict(dailydf)) * dailydf.LAI * dailydf.dayfrac1
    dailydf["gA_daily"] = np.exp(m30.predict(dailydf)) * dailydf.LAI * dailydf.dayfrac1
    
    dailydf["gpp_pred_daily"] = dailydf["amax_daily"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_daily"]))
#%%
    # tfacA = np.exp(-(dailydf.T_AIR-toptA)**2/20**2/2) / np.exp(-(25-toptA)**2/20**2/2)
    # rfacA = (dailydf.PPFD_in) / (dailydf.PPFD_in + parkA) / (800/(800+parkA))
    # dailydf["amax_daily"] = maxA*tfacA*rfacA * dailydf.LAI * dailydf.dayfrac1
    
    # tfacG = np.exp(-(dailydf.T_AIR-toptG)**2/20**2/2) / np.exp(-(25-toptG)**2/20**2/2)
    # rfacG = (dailydf.PPFD_in) / (dailydf.PPFD_in + parkG) / (800/(800+parkG))
    # dailydf["gA_daily"] = maxG*tfacG*rfacG * dailydf.LAI * dailydf.dayfrac1
    
    # dailydf["gpp_pred_daily"] = dailydf["amax_daily"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_daily"]))
    # #%%
    # # gpp_pred_daytime = k_pred*(1 - np.exp(-dailydf.cond_daytime/k_pred*slope))
    # # gpp_pred_daily = (1-dailydf.NIGHT)*gpp_pred_daytime
    #dailydf["gpp_pred_daily2"] = dailydf["amax_hourly"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_hourly"]))

    # #%%
    # dailydf["gA_daytime"] = k_pred/slope
    # dailydf["gpp_slope"] = slope
    #%%
    #%%
    gs2 = pd.merge(daily_gs,dailydf[["date","gA_daily","amax_daily","gA_hourly","amax_hourly","NIGHT","cond_daytime","vpd_daytime","P_F_QC"]],on="date",how="left")
#%%
    dfull = gs2.copy()
    dfull = dfull.loc[dfull.P_F_QC == 0].copy()
    dfull = dfull.loc[dfull.airt > 0].copy()
    dfull = dfull.loc[dfull.par > 0].copy()
    # dfull["topt"] = topt1
    # dfull["tsigma"] = sigma1
    # dfull["kSWrad"] = parK
    # dfull["gmax0"] = gmax
#%%
    # dfull["dayfrac"] = 1-dfull.NIGHT
    # dfull["kgpp"] = dfull.gA_daytime*dfull.dayfrac
    # dfull["vpd_fullday"] = 1*dfull.vpd
    # dfull["vpd"] = 1*dfull.vpd_daytime
    #%%
    site_tab.append(dfull)
#%%

site_tab = pd.concat(site_tab).reset_index()
#%%
#site_tab = pd.merge(site_tab,bif_forest[["SITE_ID","LOCATION_LAT","LOCATION_LONG"]],on="SITE_ID",how="left")
#%%
    #x1_norm = (x1-np.min(x1))/(np.max(x1)-np.min(x1))
    #x1_rescale = x1_norm*(np.max(y1)-np.min(y1)) + np.min(y1)
# plt.figure()
# plt.plot(x1-np.nanmean(x1),y1-0.5,'.')
# plt.plot([-200,200],[-0.17,0.17])
# #%%
# plt.plot(x1/521)
# plt.plot(y1)
#%%
#site_tab.to_csv("hourly_gs_data_evit75both_2mod.csv")
#%%
#dfnoon = dfday.loc[(dfday.hour == 12)].copy()
dfnoon = dfday.loc[(dfday.PPFD_in > 800)].copy()

#%%
garr = np.linspace(0,1.6,500)
# ci_arr = np.linspace(1,150,500)
# a_mm = 62*ci_arr/(ci_arr+100)
# g_mm = a_mm/(160-ci_arr)*1.6

# Amax,c0,K = 62,160,100
# g_c = garr/1.6
# Acoef = -1/g_c
# Bcoef = c0 + K + 1/g_c*Amax
# Ccoef = -Amax*c0
# aquad = (-Bcoef + np.sqrt(Bcoef**2 - 4*Acoef*Ccoef))/(2*Acoef)
#%%
def tofit1(pars):
    amax,k = pars
    gpred = amax*(1-np.exp(-dfnoon.cond/amax*k))
    return gpred-dfnoon.gpp

g_c_data = dfnoon.cond/1.6

def tofit2(pars):
    Amax,c0,K = pars
    
    Acoef = -1/g_c_data
    Bcoef = c0 + K + 1/g_c_data*Amax
    Ccoef = -Amax*c0
    
    gpred = (-Bcoef + np.sqrt(Bcoef**2 - 4*Acoef*Ccoef))/(2*Acoef)
    return gpred-dfnoon.gpp
#%%
fit0 = np.array([35,100])
myfit = scipy.optimize.least_squares(tofit1,x0=fit0,method="lm",x_scale=np.abs(fit0))
amax,k  = myfit.x
#%%
fit1 = np.array([35,160,100])
myfit = scipy.optimize.least_squares(tofit2,x0=fit1,method="lm",x_scale=np.abs(fit1))
Amax,c0,K  = myfit.x

g_c = garr/1.6
Acoef = -1/g_c
Bcoef = c0 + K + 1/g_c*Amax
Ccoef = -Amax*c0
aquad = (-Bcoef + np.sqrt(Bcoef**2 - 4*Acoef*Ccoef))/(2*Acoef)
#%%

plt.figure(dpi=200)
plt.plot(dfnoon.cond,dfnoon.gpp,'.',label="Eddy covariance"); 
plt.plot(garr,aquad,"k",linewidth=3,label="Michelis-Menten with diffusion")

plt.plot(garr,amax*(1-np.exp(-garr/amax*k)),"r--",linewidth=3,label="Exponential model")
plt.xlim(0,1.6)
plt.ylim(0,50)
plt.xlabel("Hourly canopy conductance $(mol/m^2/s)$")
plt.ylabel("Hourly GPP $(\mu mol/m^2/s)$")
plt.legend()