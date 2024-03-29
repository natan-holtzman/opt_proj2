# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:54:18 2023

@author: nholtzma
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
import scipy.optimize
import statsmodels.formula.api as smf


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
#%%
def r2_skipna(pred,obs):
    goodxy = np.isfinite(pred*obs)
    return 1 - np.mean((pred-obs)[goodxy]**2)/np.var(obs[goodxy])
#all_daily = pd.read_csv("all_yearsites_2gpp.csv",parse_dates=["date"])
all_daily = pd.read_csv("dailydata_feb22.csv",parse_dates=["date"])

#%%
site_tab = []
#%%
for site in bif_forest.SITE_ID:
    print(site)
    try:
        fname = [x for x in forest_all if site in x][0]
    except:
        continue
    #%%
    df = pd.read_csv(fname)
    df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"],format="%Y%m%d%H%M")
    df["date"] = df["TIMESTAMP_START"].dt.date
    df["hour"] = df.TIMESTAMP_START.dt.hour
    df["doy_raw"] = df.TIMESTAMP_START.dt.dayofyear
    
    df["year_raw"] = df.TIMESTAMP_START.dt.year
    df = df.loc[df.year_raw >= 2001].copy()
    #%%
    
    #%%
    if len(df) < 25*24:
        continue
    #%%
    dfull = all_daily.loc[all_daily.SITE_ID==site].copy()
    #%%
    dcount = dfull.groupby("year_new").count().reset_index()
    fullyear = dcount.year_new.loc[dcount.date > 300]
    dfull = dfull.loc[dfull.year_new.isin(fullyear)]
    #%%
    if np.max(dfull.LAI) < 0.05:
        continue
    if len(dfull) < 300:
        continue
    #%%
    
    dfull["LAI"] = np.clip(dfull["LAI"],0.05,np.inf)
   
    #%%
    df[df == -9999] = np.nan
    
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
    #df["LAI_gt50"] = (df.gpp_smooth/df.gpp_y95) > 0.67
    year_list = pd.unique(dfull.year_new)
    #%%
    if len(dfull) == 0:
        continue
    #%%
    dclim = dfull.groupby("doy").mean(numeric_only=True).reset_index()
#%%
    gpp_clim_std = np.array(dclim.LAI)/(np.nanmax(dclim.LAI))

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
    #%%
    daily_gs = dfull.loc[dfull.clim_summer].copy()
    
    daily_gs["date"] = daily_gs["date"].dt.date
   # dfull["date"] = dfull["date"].dt.date

    #%%
    df = pd.merge(df,daily_gs[["date","LAI"]],on="date",how="inner")
#    df = pd.merge(df,dfull[["date","LAI"]],on="date",how="inner")

    #%%
    dfday = df.loc[df.PPFD_in >= 5].copy()
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
    dfday = dfday.loc[dfday.year_raw >= 2001].copy()
#%%
    if len(dfday) < 25:
        continue
    #%%
    dfday["cond_norm"] = dfday.cond/dfday.LAI
    dfday["gpp_norm"] = dfday.gpp/dfday.LAI

    dfhi = dfday.loc[dfday.cond_norm > np.quantile(dfday.cond_norm,0.9)].copy()
    #%%
    himean = np.mean(dfhi.gpp_norm)
    fit0 = np.array([himean,300,himean/150,400])
    
    #%%
    def tofit(pars):
        amax1,kA,gmax1,kG = pars
        
        amax = amax1*dfday.PPFD_in/(dfday.PPFD_in + kA)
        gA = gmax1*dfday.PPFD_in/(dfday.PPFD_in + kG)

        gpp_pred = amax*(1-np.exp(-dfday.cond_norm/gA))
        z = (gpp_pred-dfday.gpp_norm)*dfday.LAI#[dfday.VPD > 1]
        return z
    
    myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))

    amax1,kA,gmax1,kG = myfit.x
    
    gA = gmax1*dfday.PPFD_in/(dfday.PPFD_in + kG)
    z1 = 1-np.exp(-dfday.cond_norm/gA)
    amax = amax1*dfday.PPFD_in/(dfday.PPFD_in + kA)
    gpp_pred_h = dfday.LAI*amax*(1-np.exp(-dfday.cond_norm/gA))
    # dfday["cga"] = dfday.cond_norm/gA
    # dfday["amax_full"] = dfday.LAI * amax
    
    # linmod_test = smf.ols("gpp ~ 0 + cga:amax_full + amax_full",data=dfday,missing='drop').fit()
    # gpp_linpred = dfday.cga*dfday.amax_full*linmod_test.params.iloc[0] + dfday.amax_full*linmod_test.params.iloc[1]
    # #%%
    # linmod_r2 = r2_skipna(gpp_linpred,dfday.gpp)
    # expmod_r2 = r2_skipna(gpp_pred_h,dfday.gpp)
    #%%
    z2 = dfday.gpp_norm/amax
    #%%
    df["gA_hourly"] = gmax1*df.PPFD_in/(df.PPFD_in + kG) * df.LAI
    df["amax_hourly"] = amax1*df.PPFD_in/(df.PPFD_in + kA) * df.LAI * (df.PPFD_in >= 5)
    df["gpp_pred_from_hourly"] = df["amax_hourly"] * (1 - np.exp(-df.cond/df["gA_hourly"]))
    #%%
    df.loc[np.isnan(df["gpp_pred_from_hourly"]),"gpp_pred_from_hourly"] = 0
    
    #%%
    daytime_avg = df.loc[df.PPFD_in >= 5].groupby("date").mean(numeric_only=True).reset_index()
    #%%
    df["day100"] = df.PPFD_in >= 5
    #%%
    dailydf = df.groupby("date").mean(numeric_only=True).reset_index()
    dailycount = df.groupby("date").count().reset_index()


    #%%
    daytime_avg["cond_daytime"] = daytime_avg.LE/44200/(daytime_avg.VPD/100)
    daytime_avg["vpd_daytime"] = daytime_avg.VPD
    daytime_avg["daytime_airt"] = daytime_avg.T_AIR
    daytime_avg["daytime_par"] = daytime_avg.PPFD_in
    #%%
    dailydf = pd.merge(dailydf,daytime_avg[["date","cond_daytime","vpd_daytime","daytime_airt","daytime_par"]],how="left",on="date")
    #%%
    dailydf["cond_daily_dayVPD"] = dailydf.LE/44200/(daytime_avg.VPD/100)

    #%%
    

    #%%
#    dailydf["dayfrac1"] = 1-dailydf["NIGHT"]
    dailydf["dayfrac1"] = dailydf["day100"]
    dailydf["gA_daily"] = gmax1*dailydf["daytime_par"]/(dailydf["daytime_par"] + kG) * dailydf.LAI * dailydf.dayfrac1
    dailydf["amax_daily"] = amax1*dailydf["daytime_par"]/(dailydf["daytime_par"] + kA) * dailydf.LAI * dailydf.dayfrac1
    dailydf["gpp_pred_daily"] = dailydf["amax_daily"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_daily"]))
    #%%
    dailydf["gpp_pred_hourly2"] = dailydf["amax_hourly"] * (1 - np.exp(-dailydf.cond_daily_dayVPD/dailydf["gA_hourly"]))

#    gs2 = pd.merge(daily_gs,dailydf[["date","gpp_interp","daytime_airt","daytime_par","NIGHT","cond_daytime","vpd_daytime","P_F_QC"]],on="date",how="left")
    gs2 = pd.merge(daily_gs,dailydf[["date","daytime_airt","daytime_par","NIGHT","cond_daytime","vpd_daytime","P_F_QC",
                                     "gA_hourly","gA_daily","amax_hourly","amax_daily","gpp_pred_from_hourly"]],on="date",how="left")
    gs2["gppR2_hourly"] = r2_skipna(gpp_pred_h,dfday.gpp)

#%%
    dfull = gs2.copy()
    
    
    site_tab.append(dfull)
#%%
site_tab = pd.concat(site_tab).reset_index()

site_tab.to_csv("data_with_daytime_feb22_noRainQC.csv")
#%%
