# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:54:18 2023

@author: nholtzma
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob

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
def find_peaks(x):
    peaklocs = np.where((x[1:-1] >= x[:-2])*(x[1:-1] >= x[2:]))[0] + 1
    return peaklocs
#if there is only one peak, we're good
#if there are two peaks, check how much lower the middle trough is

def find_valleys(x):
    valleys = np.where((x[1:-1] <= x[:-2])*(x[1:-1] <= x[2:]))[0] + 1
    return valleys

def wrap_min(x):
    minday = np.argmin(x)
    x2 = 0*x
    chunk1 = x[:minday]
    chunk2 = x[minday:]
    x2[:len(chunk2)] = 1*chunk2
    x2[len(chunk2):] = 1*chunk1
    return x2


def make_1peak(x1):
    x2 = 0*x1
    topday = np.argmax(x1)
    x2[topday] = x1[topday]
    
    for i in range(topday+1,len(x1)):
        x2[i] = min(x2[i-1],x1[i])
    for i in range(topday-1,-1,-1):
        x2[i] = min(x2[i+1],x1[i])
    return x2
#test_sites = np.array(latsort.SITE_ID)[::5]
#%%
#all_daily = pd.read_csv("all_yearsites_2gpp.csv",parse_dates=["date"])
all_daily = pd.read_csv("dailydata_aug21.csv",parse_dates=["date"])

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
    dfull["LAIdiff"] = [0] + list(np.diff(dfull.LAI))

    #gpp_clim = np.array(dclim.gpp)
    year95 = dfull.groupby("year_new").max(numeric_only=True).reset_index()

    #year95["gpp_y95"] = 1*year95["gpp_smooth"]
    year95["lai_y95"] = 1*year95["LAI"]
    dfull = pd.merge(dfull,year95[["year_new","lai_y95"]],how="left",on="year_new")
    dfull["LAI_gt50"] = dfull.LAI/dfull.lai_y95 > 0.5
    #dfull["gpp"] = np.clip((dfull.gpp_nt + dfull.gpp_dt)/2,0,np.inf)
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
    gs_starts = []
    gs_ends = []
    for year in year_list:
        dfy = dfull.loc[dfull.year_new==year].reset_index()
        topday = np.argmax(dfy.LAI)
        
        yearclim = (dfy.LAI-np.nanmin(dfy.LAI))/(np.nanmax(dfy.LAI)-np.nanmin(dfy.LAI))
        #yearclim = (dfy.LAI)/(np.nanmax(dfy.LAI))
        #yearclim = (dfy.gpp)/(np.nanmax(dfy.gpp))

#        topday = np.argmax(dfy.gpp_smooth)
        under50 = np.where(yearclim < 0.75)[0]
        #under50 = np.where(~dfy.LAI_gt50)[0]
        try:
            summer_start = under50[under50 < topday][-1] + 1
        except:
            summer_start = 0
        try:
            summer_end = under50[under50 > topday][0] -1
        except:
            summer_end = 365
        gs_starts.append(summer_start)
        gs_ends.append(summer_end)
        
    summer_df = pd.DataFrame({"year_new":year_list,
                              "summer_start":gs_starts,
                              "summer_end":gs_ends})
    dfull= pd.merge(dfull,summer_df,on="year_new",how="left")
    
    is_summer = np.array((dfull.doy >= dfull.summer_start)*(dfull.doy <= dfull.summer_end))
    
#    gpp_clim_std = np.array(dclim.LAI-np.nanmin(dclim.LAI))/(np.nanmax(dclim.LAI)-np.nanmin(dclim.LAI))
    #%%
    if len(dfull) == 0:
        continue
    #%%
    dclim = dfull.groupby("doy").mean(numeric_only=True).reset_index()
#%%
    gpp_clim_std = np.array(dclim.LAI)/(np.nanmax(dclim.LAI))
    #gpp_clim_std = np.array(dclim.LAI-np.nanmin(dclim.LAI))/(np.nanmax(dclim.LAI)-np.nanmin(dclim.LAI))

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
    #%%
  
    #%%
#    daily_gs = dfull.loc[is_summer*dfull.clim_summer].copy()
    
    daily_gs = dfull.loc[dfull.clim_summer].copy()
    
    #daily_gs = daily_gs.loc[daily_gs.LAI >= np.nanmedian(dfull.LAI)].copy()
    #daily_gs = dfull.loc[is_summer].copy()

    #daily_gs = daily_gs.loc[daily_gs.year_new >= 2001]
    
    daily_gs["date"] = daily_gs["date"].dt.date
    #%%
    
    
    #%%
    df = pd.merge(df,daily_gs[["date","LAI","lai_y95"]],on="date",how="inner")
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
    dfday = dfday.loc[dfday.year_raw >= 2001].copy()
#%%
    if len(dfday) < 25:
        continue
#%% 
    
    #%%
    dfday["cond_norm"] = dfday.cond/dfday.LAI
    dfday["gpp_norm"] = dfday.gpp/dfday.LAI

    dfhi = dfday.loc[dfday.cond_norm > np.quantile(dfday.cond_norm,0.9)].copy()
    #%%
    
    #%%
    gpp2 = np.array(df.gpp)
    for hi in range(24):
        z = gpp2[hi::24]*1
        z2 = np.interp(np.arange(len(z)), np.arange(len(z))[np.isfinite(z)], z[np.isfinite(z)])
        gpp2[hi::24] = z2
    
    df["gpp_interp"] = gpp2
    
    #%%
#    daytime_avg = df.loc[df.NIGHT==0].groupby("date").mean(numeric_only=True).reset_index()
    daytime_avg = df.loc[df.PPFD_in >= 100].groupby("date").mean(numeric_only=True).reset_index()
    #%%
    df["day100"] = df.PPFD_in >= 100
    #%%
    dailydf = df.groupby("date").mean(numeric_only=True).reset_index()
    #%%
    dailydf["cond_daily_dayVPD"] = dailydf.LE/44200/(daytime_avg.VPD/100)

    #%%
    dailydf["cond_daytime"] = daytime_avg.LE/44200/(daytime_avg.VPD/100)
    dailydf["vpd_daytime"] = daytime_avg.VPD
    dailydf["daytime_airt"] = daytime_avg.T_AIR
    dailydf["daytime_par"] = daytime_avg.PPFD_in

    #%%
#    dailydf["dayfrac1"] = 1-dailydf["NIGHT"]
    dailydf["dayfrac1"] = dailydf["day100"]

    
    #%%
    gs2 = pd.merge(daily_gs,dailydf[["date","gpp_interp","daytime_airt","daytime_par","NIGHT","cond_daytime","vpd_daytime","P_F_QC"]],on="date",how="left")
#%%
    dfull = gs2.copy()
    dfull = dfull.loc[dfull.P_F_QC == 0].copy()
    
    site_tab.append(dfull)
#%%

site_tab = pd.concat(site_tab).reset_index()
#%%

#%%
site_tab.to_csv("data_with_daytime_aug21.csv")
