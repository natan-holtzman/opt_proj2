# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:46:32 2022

@author: nholtzma
"""


import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import glob
import datetime
#%%
do_bif = 0
if do_bif:
    biftab = pd.read_csv(r"FLX_AA-Flx_BIF_ALL_20200501\FLX_AA-Flx_BIF_HH_20200501.csv")
    groups_to_keep = ["GRP_CLIM_AVG","GRP_HEADER","GRP_IGBP","GRP_LOCATION","GRP_SITE_CHAR","GRP_DOM_DIST_MGMT"]#,"GRP_LAI","GRP_ROOT_DEPTH","SOIL_TEX","SOIL_DEPTH"]
    biftab = biftab.loc[biftab.VARIABLE_GROUP.isin(groups_to_keep)]
    bif2 = biftab.pivot_table(index='SITE_ID',columns="VARIABLE",values="DATAVALUE",aggfunc="first")
    bif2.to_csv("fn2015_bif_tab_h.csv")
#%%
bif_data = pd.read_csv("../optimality_proj/fn2015_bif_tab.csv")
#bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF"])]
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH","CRO"])]
#metadata = pd.read_csv(r"C:\Users\nholtzma\Downloads\fluxnet_site_info_all.csv")
#bif_forest = bif_forest.loc[~bif_forest.SITE_ID.isin(["IT-CA1","IT-CA3"])]
all_daily = glob.glob("../optimality_proj/daily_data\*.csv")
forest_daily = [x for x in all_daily if x.split("\\")[-1].split('_')[1] in list(bif_forest.SITE_ID)]
#%%
#all_evi = pd.read_csv("evi_ndvi_allsites_sq.csv")

#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%

def cor_skipna2(x,y):
    goodxy = np.isfinite(x*y)
    return np.corrcoef(x[goodxy],y[goodxy])[0,1]


zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf
def getcols(df,varname):
    c1 = [x for x in df.columns if x.startswith(varname+"_") or x==varname]
    return [x for x in c1 if not x.endswith("QC")]

def meancols(df,varname):
    sel_cols = getcols(df,varname)
    if len(sel_cols) == 0:
        return np.nan*np.zeros(len(df))
    col_count = df[sel_cols].count()
    best_col = sel_cols[np.argmax(col_count)]
    return df[best_col]


def lastcols(df,varname):
    sel_cols = getcols(df,varname)
    if len(sel_cols) == 0:
        return np.nan*np.zeros(len(df)), "none"
    best_col = sel_cols[-1]
    return df[best_col], best_col

def fill_na(x):
    return np.interp(np.arange(len(x)), np.arange(len(x))[np.isfinite(x)], x[np.isfinite(x)])

def fill_na2(x,y):
    x2 = 1*x
    x2[np.isnan(x2)] = 1*y[np.isnan(x2)]
    return x2
#%%
#fname = [x for x in forest_daily if site_id in x][0]
def prepare_df(fname, site_id, bif_forest):
    #%%
    df = pd.read_csv(fname,parse_dates=["TIMESTAMP"])
    df[df==-9999] = np.nan
    latdeg = bif_forest.loc[bif_forest.SITE_ID==site_id].LOCATION_LAT.iloc[0]
    
    df["date"] = df["TIMESTAMP"].dt.date
    df["hour"] = df["TIMESTAMP"].dt.hour
    df["doy"] = df["TIMESTAMP"].dt.dayofyear
    df["month"] = df["TIMESTAMP"].dt.month

    site_id = fname.split("\\")[-1].split('_')[1]
    #print(site_id)
    
    df["year"] = df["TIMESTAMP"].dt.year
    df["year_new"] = 1*df["year"]
    
    df["doy_new"] = 1*df.doy
    if latdeg < 0:
        df["doy_new"] = (df["TIMESTAMP"] + datetime.timedelta(days=182)).dt.dayofyear
        df["year_new"] = (df["TIMESTAMP"] + datetime.timedelta(days=182)).dt.year

    #%%
    laifile = "../optimality_proj/lai_csv/lai_csv/" + "_".join(site_id.split("-")) + "_LAI_FLX15.csv"
    try:
        laidf = pd.read_csv(laifile,parse_dates=["Time"])
        df2 = pd.merge(df,laidf,left_on = "TIMESTAMP",right_on="Time",how='left')
        lai_all = np.array(df2.LAI)
        lai_all = np.interp(np.arange(len(lai_all)),
                            np.arange(len(lai_all))[np.isfinite(lai_all)],
                            lai_all[np.isfinite(lai_all)])
        df2 = None
    except:
        lai_all = np.zeros(len(df))
    df["LAI"] = lai_all

    #%%

    par_summer = np.array(meancols(df,'SW_IN_F'))
    potpar = np.array(meancols(df,'SW_IN_POT'))
    #%%
    airt_summer = np.array(meancols(df,"TA"))
    #rh_summer = np.array(meancols(df,"RH"))/100
    SatVP = 6.1094*np.exp(17.625*airt_summer/ (airt_summer+ 243.04))/10  #kpa
    vpd_summer =  np.array(meancols(df,"VPD"))/10  #SatVP*(1-rh_summer)
    
    et_summer = np.array(df['LE_F_MDS']) / 44200 

    et_qc = np.array(df.LE_F_MDS_QC)
    et_summer[et_qc < 0.9] = np.nan
    #%%
    
    le_25 = np.array(df['LE_CORR_25']) #/ 44200 
    le_75 = np.array(df['LE_CORR_75']) #/ 44200 
    #et_summer[np.isnan(le_25*le_75)] = np.nan
    
    myrn = np.array(meancols(df,"NETRAD"))
     
    myg = np.array(meancols(df,"G")) #-myG
    if np.mean(np.isfinite(myg)) == 0:
        myg = 0
    #vpd_summer = np.array(df["VPD_F"])/10#*10 #hPa to kPa
    
    vpd_summer[vpd_summer < 0.1] = np.nan
    
    et_summer[et_summer <= 0] = np.nan
    #et_summer[np.isnan(etunc_summer)] = np.nan
    #%%
    try:
        ground_heat = np.array(df["G_F_MDS"])
    #ground_heat[np.isnan(ground_heat)] = 0
    except KeyError:
        ground_heat = 0.1*myrn
    if np.mean(np.isfinite(ground_heat)) < 0.5:
        ground_heat = 0.1*myrn

    rain_summer = np.array(df["P_F"])
    #%%
    if np.sum(np.isfinite(et_summer)) < (25):
        print("Not enough ET")
        return "Not enough data"
        
    #%%
    df['etqc'] = et_summer
    
    my_clim = df.groupby("doy_new").mean(numeric_only=True)
    
    gpp_clim = np.array(1*my_clim["GPP_DT_VUT_REF"] + 1*my_clim["GPP_NT_VUT_REF"])/2
    
    #gpp_clim_std = gpp_clim - np.nanmin(gpp_clim)
    
    gpp_adjoin = fill_na(np.tile(gpp_clim,3))
    
    gpp_clim_smooth = np.zeros(len(gpp_adjoin))
    swidth = 14
    
    for i in range(swidth,len(gpp_adjoin)-swidth):
        gpp_clim_smooth[i] = np.nanmean(gpp_adjoin[i-swidth:i+swidth+1])

    gpp_clim_smooth[:swidth] = np.mean(gpp_clim[:swidth])
    gpp_clim_smooth[-swidth:] = np.mean(gpp_clim[-swidth:])  
    
    gpp_summer = np.array(1*df["GPP_DT_VUT_REF"] + 1*df["GPP_NT_VUT_REF"])/2
    #gpp_summer_nt = np.array(df["GPP_NT_VUT_REF"])

    #airt_summer[airt_summer < 0] = np.nan
    gpp_summer[gpp_summer < 0] = np.nan
    
    nee_qc = np.array(df.NEE_VUT_REF_QC)
    gpp_summer[nee_qc < 0.5] = np.nan

#%%
    lai_clim = np.array(my_clim.LAI)
#    lai_clim_std = (lai_clim - np.min(lai_clim))/(np.max(lai_clim) - np.min(lai_clim))
    lai_clim_std = lai_clim / np.max(lai_clim)
    
    topday = np.argmax(lai_clim_std)
    under50 = np.where(lai_clim_std < 0.8)[0]
    #%%
    try:
        summer_start = under50[under50 < topday][-1]
    except IndexError:
        summer_start = 0
    try:
        summer_end = under50[under50 > topday][0]
    except IndexError:
        summer_end = 365
    df["summer_start"] = summer_start
    df["summer_end"] = summer_end
    
    #gpp_smooth[:swidth] = np.mean(gpp_summer[:swidth])
    #gpp_smooth[-swidth:] = np.mean(gpp_summer[-swidth:])
    #%%
    df["gpp_smooth"] = 0
    
#    year95 = df.groupby("year_new").quantile(0.95,numeric_only=True).reset_index()
    year95 = df.groupby("year_new").max(numeric_only=True).reset_index()

    year95["gpp_y95"] = 1*year95["gpp_smooth"]
    year95["lai_y95"] = 1*year95["LAI"]
#%%
#    yearMin = df.groupby("year_new").quantile(0.05,numeric_only=True).reset_index()
    yearMin = df.groupby("year_new").min(numeric_only=True).reset_index()

    yearMin["lai_ymin"] = 1*yearMin["LAI"]
    yearMin["gpp_ymin"] = 1*yearMin["gpp_smooth"]

    #%%
    df = pd.merge(df,year95[["year_new","lai_y95","gpp_y95"]],how="left",on="year_new")
    
    df = pd.merge(df,yearMin[["year_new","lai_ymin","gpp_ymin"]],how="left",on="year_new")
    
    #%%
    plot_gs = 0
    if plot_gs:
        #%%
        plt.figure()
        plt.plot(gpp_clim)
        plt.plot(gpp_clim_smooth)
        #plt.axvspan(summer_start,summer_end,color="green",alpha=0.33)
        plt.twinx()
        plt.plot(my_clim.SW_IN_POT,'r')
        plt.twinx()
        plt.plot(my_clim.LAIclim,'k')

    #%%

    my_clim = my_clim.reset_index()
    my_clim["P_F_c"] = fill_na(np.array(my_clim.P_F))
#    my_clim["LE_all_c"] = fill_na(np.array(my_clim.LE_F_MDS))
    my_clim["LE_all_c"] = fill_na(np.array(my_clim.etqc))

    dfm = pd.merge(df,my_clim[["doy_new","P_F_c","LE_all_c"]],on="doy_new",how="left")        
    
    p_in = fill_na2(np.array(df.P_F),np.array(dfm.P_F_c))
    et_out = fill_na2(et_summer * 18/1000 * 60*60*24,np.array(dfm["LE_all_c"] * 18/1000 * 60*60*24))
    doy_summer = np.array(df["doy_new"])
    #%%
    df["et_wqc"] = et_summer
    
    smc_summer, smc_name = lastcols(df,'SWC')
    smc_summer = np.array(smc_summer)
    

    df["smc"] = smc_summer
    #%%
    
    
   # ground_heat = 0
    
    SatVP = 6.1094*np.exp(17.625*airt_summer/ (airt_summer+ 243.04))/10  #kpa
    
    wsarr = np.array(meancols(df,'WS'))
    
    #wsarr[wsarr == 0] = 0.025
    # myga_old = 0.41**2*wsarr / (np.log(2.4/35))**2
    ustar = np.array(meancols(df,"USTAR"))
    #myga = (wsarr/ustar**2 + 6.2*ustar**(-2/3))**-1
    myga = ustar**2/wsarr
    
    #myga = 1/(1/myga + 6.2*ustar**-0.667)
    
    lambda0 = 2.26*10**6
    sV = 0.04145*np.exp(0.06088*airt_summer) #in kpa
    gammaV = 100*1005/(lambda0*0.622) #in kpa
    
    petVnum = (sV*(myrn-ground_heat) + 1.225*1000*vpd_summer*myga)*(myrn > 0) #/(sV+gammaV*(1+myga[i]/(gmax*condS*mylai[i])))  #kg/s/m2 
    
    g_ratio = (petVnum / (et_summer*44200) - sV)/gammaV - 1
    inv2 = myga/g_ratio
    
    inv2_stp = inv2/0.0224
    
    patm_summer =  np.array(meancols(df,"PA"))
    patm_summer[np.isnan(patm_summer)] = 101.325
    
    gasvol_fac = (airt_summer + 273.15)/(25+273.15) * 101.325/patm_summer
    
    inv2_varTP = inv2/(22.4*gasvol_fac/1000)
    
    daily_cond = inv2_varTP
    daily_cond[daily_cond > 2] = np.nan
    daily_cond[daily_cond <= 0] = np.nan
    
    
    pet = petVnum/(sV+gammaV)

    #%%
    if np.sum(np.isfinite(gpp_summer)) < (25):
        print("Not enough GPP")
        return "Not enough data"

    #%%
    # houri = 12
    # deg_noon = 360 / 365 * (doy_summer + houri / 24 + 10);
    # decd = -23.44*np.cos(deg_noon*np.pi/180)
    # lhad = (houri-12)*15
    
    # cosz = (np.sin(latdeg*np.pi/180) * np.sin(decd*np.pi/180) + 
    #         np.cos(latdeg*np.pi/180) * np.cos(decd*np.pi/180) *
    #         np.cos(lhad*np.pi/180))
      
    #%%
    petVnum[petVnum==0] = np.nan
    gpp_summer = np.array(gpp_summer)
    
    rain_prev = 0*rain_summer
    rain_prev[1:] = rain_summer[:-1]
    #%%
   
#%%
    df_to_fit = pd.DataFrame({"date":df.date,"airt":airt_summer,"year":df.year,"year_new":df.year_new,
                              "par":par_summer,#"cosz":cosz,
                              "potpar":potpar,
                              
                              "cond":daily_cond,"gpp":gpp_summer,
                              "doy":doy_summer,"vpd":vpd_summer,
                              "doy_raw":np.array(df.doy),
                              "ET":et_summer,"ET_qc":et_qc,
                              
                              "rain":rain_summer,
                              "rain_prev":rain_prev,
                              "rain_qc":df["P_F_QC"],
                              "LAI":lai_all,
                              "smc":smc_summer,
                              
                              "gpp_dt":df["GPP_DT_VUT_REF"],
                              "gpp_nt":df["GPP_NT_VUT_REF"]

                             
                              })
    
    df_to_fit["netrad"] = myrn
    # df_to_fit["gs_netrad"] = np.nanmean(myrn[is_summer])
    df_to_fit["SITE_ID"] = site_id
#%%

    return df_to_fit 
#%%
#%%
#%%
#width = 1
zsoil_mm_base = 1000
zsoil_mol = zsoil_mm_base*1000/18 #now depth in cm from BIF
gammaV = 0.07149361181458612
width= np.inf
gmax = np.inf
#%%

rain_dict = {}
year_tau_dict = {}
all_results = []
site_result = {}
#%%
for fname in forest_daily:#[forest_daily[x] for x in [70,76]]:
#%%
    site_id = fname.split("\\")[-1].split('_')[1]
    print(site_id)
    df_res = prepare_df(fname, site_id, bif_forest)
    #%%
    if type(df_res) == str:
        site_result[site_id] = df_res
        continue
    #%%
    df_to_fit = df_res
    #rain_dict[site_id] = rain_res
    #%%
    if len(df_to_fit) < 25:
        print("Not enough data")
        site_result[site_id] = "Not enough data"

        continue
    #%%
    # if sum(np.isfinite(df_to_fit.gpp_qc)) < 25:
    #     print("Not enough data")
    #     site_result[site_id] = "Not enough data"
    #     continue
    #%%

    #%%
    #dfi = df_to_fit.loc[df_to_fit.waterbal < np.nanmedian(df_to_fit.waterbal)].copy()

   # dfi = df_to_fit.loc[(df_to_fit.doy >= topday)].copy()

    #%%    
    all_results.append(df_to_fit)
    #%%
all_results = pd.concat(all_results)
all_results.to_csv("dailydata_aug26.csv")
#%%
# sites = []
# years = []
# rains = []
# for x in rain_dict.keys():
#     ri = rain_dict[x][0]
#     sites.append(np.array([x]*len(ri)))
#     years.append(rain_dict[x][1])
#     rains.append(ri)
# #%%
# raindf = pd.DataFrame({"SITE_ID":np.concatenate(sites),
#                       "year":np.concatenate(years),
#                       "rain_mm":np.concatenate(rains)})
# raindf.to_csv("rain_80lai_climseas_april12.csv")
#%%
# di = 250
# plt.plot(daily_cond,
#          1/44200*(sV[di]*(myrn[di]-ground_heat[di]) + 1.225*1000*vpd_summer[di]*myga[di])/(sV[di] + gammaV*(1+ myga[di]/(daily_cond*(22.4*gasvol_fac[di]/1000)))),'.'); 
# plt.plot([0,0.6],[0,0.6*vpd_summer[di]/100])
# #%%
# di = 250
# plt.plot(vpd_summer,
#          1/44200*(sV[di]*(myrn[di]-ground_heat[di]) + 1.225*1000*vpd_summer*myga[di])/(sV[di] + gammaV*(1+ myga[di]/(daily_cond[di]*(22.4*gasvol_fac[di]/1000)))),'.'); 
# plt.plot([0,3.5],[0,3.5* daily_cond[di]/100])
# #%%
# et_fake1 = (sV*(np.nanmean(myrn)-ground_heat) + 1.225*1000*vpd_summer*myga)*(myrn > 0)/(sV + gammaV*(1+ myga/(daily_cond*(22.4*gasvol_fac/1000))))
# et_fake2 = (sV*(np.nanmean(myrn)-ground_heat) + 1.225*1000*vpd_summer*myga)*(myrn > 0)/(sV + gammaV*(1+ myga/(0.025*smc_summer*lai_all/np.nanmean(lai_all)*(22.4*gasvol_fac/1000))))
# et_fake3 = (sV*(myrn-ground_heat) + 1.225*1000*vpd_summer*myga)*(myrn > 0)/(sV + gammaV*(1+ myga/(0.025*smc_summer*lai_all/np.nanmean(lai_all)*(22.4*gasvol_fac/1000))))