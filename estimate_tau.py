# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:46:32 2022

@author: nholtzma
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.optimize
import statsmodels.formula.api as smf

import matplotlib as mpl
#%%
do_bif = 0
if do_bif:
    biftab = pd.read_excel(r"C:\Users\nholtzma\Downloads\fluxnet2015\FLX_AA-Flx_BIF_ALL_20200501\FLX_AA-Flx_BIF_DD_20200501.xlsx")
    groups_to_keep = ["GRP_CLIM_AVG","GRP_HEADER","GRP_IGBP","GRP_LOCATION","GRP_SITE_CHAR"]#,"GRP_LAI","GRP_ROOT_DEPTH","SOIL_TEX","SOIL_DEPTH"]
    biftab = biftab.loc[biftab.VARIABLE_GROUP.isin(groups_to_keep)]
    bif2 = biftab.pivot_table(index='SITE_ID',columns="VARIABLE",values="DATAVALUE",aggfunc="first")
    bif2.to_csv("fn2015_bif_tab.csv")
#%%
def cor_skipna(x,y):
    goodxy = np.isfinite(x*y)
    try:
        return scipy.stats.pearsonr(x[goodxy],y[goodxy])
    except:
        return (np.nan,np.nan)
def rho_skipna(x,y):
    goodxy = np.isfinite(x*y)
    return scipy.stats.spearmanr(x[goodxy],y[goodxy])

def r2_skipna(pred,obs):
    goodxy = np.isfinite(pred*obs)
    return 1 - np.mean((pred-obs)[goodxy]**2)/np.var(obs[goodxy])
#%%
bif_data = pd.read_csv("../optimality_proj/fn2015_bif_tab_h.csv")
bif_forest = bif_data.loc[bif_data.IGBP.isin(["MF","ENF","EBF","DBF","DNF","GRA","SAV","WSA","OSH","CSH","CRO"])].copy()
metadata = pd.read_csv("../optimality_proj/fluxnet_site_info_all.csv")
#%%
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7

plt.rcParams['font.size']=18
plt.rcParams["mathtext.default"] = "sf"

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
#%%
mol_s_to_mm_day = 1*18/1000*24*60*60
#%%

rain_dict = {}
year_tau_dict = {}
site_result = {}

#%%
bigyear = pd.read_csv("data_with_daytime_aug28b.csv")

#bigyear = pd.read_csv("data_with_daytime_sept13.csv")

simple_biomes = {"SAV":"Savanna",
                 "WSA":"Savanna",
                 "CSH":"Shrubland",
                 "OSH":"Shrubland",
              "EBF":"Evergreen broadleaf forest",
              "ENF":"Evergreen needleleaf forest",
              "GRA":"Grassland",
              "DBF":"Deciduous broadleaf forest",
              "MF":"Mixed forest",
              "CRO":"Crop"
              }
biome_list = ["Evergreen needleleaf forest", "Mixed forest", "Deciduous broadleaf forest", "Evergreen broadleaf forest",
              "Grassland","Shrubland","Savanna","Crop"]

#df_in["combined_biome"] = [simple_biomes[x] for x in df_in["IGBP"]]

bigyear = pd.merge(bigyear,bif_forest,on="SITE_ID",how='left')

bigyear["combined_biome"] = [simple_biomes[x] for x in bigyear["IGBP"]]

#bigyear = bigyear.loc[bigyear.year >= 2001].copy()
#%%
fullyear = pd.read_csv("dailydata_aug21.csv")
#%%
site_message = []
site_dd_limit = []
#%%
all_results = []
ddlist = []
all_results0 = []
#%%
for site_id in pd.unique(bigyear.SITE_ID):
    #%%
    #if site_id=="ZM-Mon":
    #    continue
#%%
    #if not site_id.startswith("US"):
    #    continue

#%%
    print(site_id)
    #dfgpp = df_in.loc[df_in.SITE_ID==site_id].copy()
    dfull = bigyear.loc[bigyear.SITE_ID==site_id].copy()
    dyear = fullyear.loc[fullyear.SITE_ID==site_id].copy()
    dfull.loc[dfull.gpp <= 0,"ET"] = np.nan
    
    dfull["Aridity"] = np.nanmean(dyear.netrad) / (np.nanmean(dyear.rain) / (18/1000 * 60*60*24) * 44200)
    dfull["Aridity_gs"] = np.nanmean(dfull.netrad) / (np.nanmean(dfull.rain) / (18/1000 * 60*60*24) * 44200)
    dfull["map_data"] = np.nanmean(dyear.rain)
    dfull["mat_data"] = np.nanmean(dyear.airt)
    #%%
    #dfull = pd.merge(dfull,dyear[["date","gpp_dt","gpp_nt"]],on="date",how="left")

    #%%
    z1 = np.array(dyear.rain)
    y1 = np.array(dyear.year_new)
    L1 = np.array(dyear.LAI)
    GPP1 = np.clip(np.array(dyear.gpp),0,np.inf)

    rain_days = np.array([0] + list(np.where(z1 > 0)[0]) + [len(z1)])
    ddlenI = np.diff(rain_days)#-1
    ddlenY = y1[rain_days[:-1]]
    ymax = [np.max(ddlenI[ddlenY==y]) for y in pd.unique(ddlenY)]
    dfull["fullyear_dmax"] = np.mean(ymax)
    dfull["fullyear_dmean"] = np.sum(ddlenI**2)/np.sum(ddlenI)
    
    dfull["fullyear_prain"] = np.mean(z1 > 0)

    #%%
    if len(dfull) < 25:
        site_message.append("Not enough data")
        continue
    #dcount = dfull.groupby("year_new").count().reset_index()
    #fullyear = dcount.year_new.loc[dcount.combined_biome > 300]
   # dfull = dfull.loc[dfull.year_new.isin(fullyear)]
    #%%
    if np.max(dfull.LAI) < 0.05:
        site_message.append("No LAI data")
        continue
    
    #%%
    dfull["LAI"] = np.clip(dfull["LAI"],0.05,np.inf)

    dfull["is_summer"] = True #(dfull.doy >= summer_start)*(dfull.doy <= summer_end)

    dfull["dayfrac"] = 1-dfull.NIGHT
    #dfull["kgpp"] = dfull.gA_daily#*dfull.dayfrac
    dfull["vpd_fullday"] = 1*dfull.vpd
    dfull["vpd"] = np.clip(dfull.vpd_daytime,0.1,np.inf)
    #dfull["vpd"] = np.clip(dfull.vpd,0.1,np.inf)


    dfull["cond2"] = dfull.ET/np.clip(dfull.vpd_daytime,0.1,np.inf)*100
    #%%
    # dfull.loc[dfull.rain != 0, "gpp_pred_daily"] = np.nan
    # dfull.loc[dfull.rain_prev != 0, "gpp_pred_daily"] = np.nan
    # dfull.loc[dfull.vpd <= 0.5 , "gpp_pred_daily"] = np.nan
    
    # dfull.loc[dfull.rain != 0, "gpp_pred_hourly"] = np.nan
    # dfull.loc[dfull.rain_prev != 0, "gpp_pred_hourly"] = np.nan
    # dfull.loc[dfull.vpd <= 0.5 , "gpp_pred_hourly"] = np.nan
    
    #%%
    dfull["gpp_assess"] = 1*dfull.gpp
    dfull.loc[dfull.rain != 0, "gpp_assess"] = np.nan
    dfull.loc[dfull.rain_prev != 0, "gpp_assess"] = np.nan
    dfull.loc[dfull.vpd <= 0.5 , "gpp_assess"] = np.nan
    #%%
    
    # def sfun(x):
    #     y = x*dfull.amax_hourly*(1-np.exp(-dfull.cond2/dfull.gA_hourly/x))
    #     return -r2_skipna(y,dfull.gpp_assess)
    # sres = scipy.optimize.minimize_scalar(sfun, bounds = [0.5,2])
        
    # gfac = sres.x
    # dfull["gfac"] = gfac
    # dfull.amax_hourly *= gfac
    # dfull.gA_hourly *= gfac
    
    dfull["gpp_pred_hourly"] = dfull.amax_hourly*(1-np.exp(-dfull.cond2/dfull.gA_hourly))
    dfull["gpp_pred_daily"] = dfull.amax_daily*(1-np.exp(-dfull.cond2/dfull.gA_daily))
    
    #dfull["kgpp"] = dfull.gA_hourly
    dfull["kgpp"] = dfull.gA_daily

    #%%
    dfull["gppR2_exp_daily"] = r2_skipna(dfull.gpp_pred_daily,dfull.gpp_assess)
    dfull["gppR2_exp_hourly"] = r2_skipna(dfull.gpp_pred_hourly,dfull.gpp_assess)
    #dfull["gppR2_exp"] = dfull["gppR2_exp_hourly"]
    dfull["gppR2_exp"] = dfull["gppR2_exp_daily"]

    # if r2_skipna(dfull.gpp_pred_d2,dfull.gpp) < 0:
    #     site_message.append("GPP model did not fit")
    #     continue
    #%%
    # sret = dfull.ET**2 / dfull.vpd / dfull.kgpp
    # sret[np.isnan(dfull.gpp_assess)] = np.nan
    # sret[dfull.par < 100] = np.nan
    # sret = np.array(sret)
    # smcarr = np.array(dfull.smc)
    # strans = np.interp(smcarr, np.sort(smcarr[np.isfinite(sret*smcarr)]), np.sort(sret[np.isfinite(sret*smcarr)]))
    # etret = np.sqrt(strans * dfull.vpd * dfull.kgpp)
    #%%
    dfGS = dfull.loc[dfull.is_summer].copy()
    dfull["gsrain_mean"] = np.mean(dfGS.rain)
    
    dfull["gs_prain"] = np.mean(dfGS.rain > 0)

    all_results0.append(dfull[["SITE_ID","gppR2_exp_daily","gppR2_exp_hourly"]])


#    dfGS = dfull.copy()
    #dfGS["cond_per_LAI"] = dfGS.cond/dfGS.LAI
    #cl75 =  np.nanquantile(dfGS["cond_per_LAI"],0.75)
#%%
    ddreg_fixed = []

    et_over_dd = []

    ymaxes0 = []
    ymeans0 = []
    
    ddlabel = []
    ddii = 0
    
    grec = []
    frec = []
    et_plain = []
    vpd_plain = []
    
    etcum = []
    ddyears = []
    smclist = []
    
    ddall = 0
    
    #rain_by_year = []
    
    smc_start = []
    smc_end = []
    smc_avg = []
    lai_avg = []
    vpd_avg = []
    et_avg = []
    et_init = []
    is_limited = []
    dd_doy = []
    
    individual_slopes = []
    dd_dates = []
    #%%
    for y0 in pd.unique(dfGS.year_new):
    #%%
        dfy = dfGS.loc[dfGS.year_new==y0].copy()
        if np.sum(np.isfinite(dfy.ET)) < 1:
            continue
        
        doy_indata = np.array(dfy.doy)
        vpd_arr = np.clip(np.array(dfy.vpd),0.1,np.inf)/100
        # if np.sum(np.isfinite(vpd_arr)) < 25:
        #     continue
        vpd_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(vpd_arr)],
                            vpd_arr[np.isfinite(vpd_arr)])
        k_mm_day = np.array(dfy.kgpp)*mol_s_to_mm_day #* np.array(dfy.gpp/dfy.gpp_pred)
        rain_arr = np.array(dfy.rain)
        #seaslens.append(len(rain_arr))
        #%%
    #et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        
        et_mmday = np.array(dfy.ET)*mol_s_to_mm_day
        
        #et_mmday = np.array(dfy.cond)*mol_s_to_mm_day
        
        et_mmday_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(et_mmday)],
                            et_mmday[np.isfinite(et_mmday)])
        
        
        #et_mmday[dfy.airt < 10] = np.nan
        et_mmday[dfy.par < 100] = np.nan

        et_mmday[dfy.vpd <= 0.5] = np.nan
        #et_mmday[dfy.ET_qc < 0.5] = np.nan
        #%%
        # rainacc = rain_arr[0]
        # rain_days = [0]
        # for dayi in range(1,len(rain_arr)):
        #     rainacc += rain_arr[dayi]
        #     if rainacc > 5:
        #         rain_days.append(dayi)
        #         rainacc = 0
        # rain_days = np.array(rain_days + [len(rain_arr)])
#%%
        rain_days = np.array([0] + list(np.where(rain_arr > 0)[0]) + [len(rain_arr)])
        ddgood = np.where(np.diff(rain_days) >= 5)[0] #7
        
        ddall += len(ddgood)
        
        ddstart = rain_days[ddgood]+2
        ddend = rain_days[ddgood+1]
    
        etnorm = et_mmday**2 / (vpd_interp*k_mm_day)
        
        
        doyY = np.array(dfy.doy)
    #%%
        #rain_days = np.array([0] + list(np.where(rain_arr > 0)[0]) + [len(rain_arr)])

        dd0= np.diff(rain_days) #- 1
        ymaxes0.append(np.max(dd0))
        #ymeans0.append(np.mean(dd0[dd0 >= 2]))
        ymeans0.append(np.sum(dd0**2)/np.sum(dd0))
        #rain_by_year.append(dd0)
    #%%
        #tau_with_unc = []
        #winit_with_unc = []
        for ddi in range(len(ddstart)):
            starti = ddstart[ddi]
            #starti = max(ddstart[ddi],ddend[ddi] - 50) 

            endi = ddend[ddi]
            #endi = min(starti+20,ddend[ddi])
            f_of_t = (vpd_arr*k_mm_day)[starti:endi]
            g_of_t = np.array([0] + list(np.cumsum(np.sqrt(f_of_t))))[:-1]
            
            doyDD = doyY[starti:endi]
            yfull = etnorm[starti:endi]#[:20]
                        
            etsel = et_mmday_interp[starti:endi]#[:20]
            rainsel =  rain_arr[starti:endi]
            if np.sum(np.isfinite(yfull)) >= 3: # and np.mean(np.isfinite(yfull)) >= 0.5:
                smc_avg.append(np.mean(dfy.smc.iloc[starti:endi]))

                etcumDD = np.array([0] + list(np.cumsum(etsel-rainsel)))[:-1]
                #etcumDD = np.cumsum(etsel-rainsel)
                #etcumDD -= etcumDD[0]
                rDD = sm.OLS(yfull,sm.add_constant(etcumDD),missing='drop').fit()
                #et_itself = et_mmday[starti:endi] / vpd_arr[starti:endi]
                #timecor = cor_skipna(et_itself, np.arange(len(et_itself)))
                #if timecor[0] < 0 and timecor[1] < 0.05:

                if rDD.params[1] < 0:
                #if True: #rDD.params[1] < 0:
                    individual_slopes.append([rDD.params[1]]*len(yfull))
                    ddlabel.append([ddii]*len(yfull))
                    ddyears.append([y0]*len(yfull))
                    dd_doy.append(dfy.doy.iloc[starti:endi])
                    dd_dates.append(dfy.date.iloc[starti:endi])

                    
                    frec.append(f_of_t)
                    grec.append(g_of_t)
                    vpd_plain.append(vpd_arr[starti:endi])
                    et_plain.append(et_mmday[starti:endi])
                    smclist.append(dfy.smc.iloc[starti:endi])
                    
                    etcum.append(etcumDD)
                    
                    yI = yfull - np.nanmean(yfull)
                    xI = etcumDD - np.mean(etcumDD[np.isfinite(yfull)])
                    
                    et_over_dd.append(yI)
                    ddreg_fixed.append(xI)
                    is_limited.append(1)
                    
                    ddii += 1
                else:
                    is_limited.append(0)
    #%%
    if ddall < 3:
        site_message.append("Not enough dd")
        continue
    #%% 
    if len(ddreg_fixed) < 3:
        site_message.append("Not enough water limitation")
        continue
#     #%%
#%%    
    row0 = np.concatenate(ddreg_fixed)
    #row0[np.abs(row0) > np.nanstd(row0)*3] = np.nan

    et_topred = np.concatenate(et_over_dd)
    et_topred[np.abs(et_topred) > np.nanstd(et_topred)*3] = np.nan
    #et_topred[np.abs(row0) > np.nanstd(row0[np.isfinite(et_topred)])*3] = np.nan

    # if np.sum(np.isfinite(et_topred*row0)) < 10:
    #     continue
    #%%
    r1= sm.OLS(et_topred,row0,missing='drop').fit()
    # if r1.pvalues[0] > 0.05 or r1.params[0] > 0:
    #     continue
    dfull["reg_npoints"] = np.sum(np.isfinite(et_topred))
    dfull["reg_ndd"] = len(et_over_dd)
    dfull["reg_pval"] = r1.pvalues[0]
    #dfull["tau_med"] = -2/np.median(slopelist)
    #%%
    dfull["tau_ddreg0"] = -2/r1.params[0]
    dfull["tau_ddreg0_lo"] = -2/(r1.params[0] - 2*r1.bse[0])
    dfull["tau_ddreg0_hi"] = -2/(r1.params[0] + 2*r1.bse[0])
    #%%
    dfull["gslen_annual"] = np.max(dfull.doy) - np.min(dfull.doy) # np.mean(seaslens)
    dfull["tau0_rel_err"] = -r1.bse[0]/r1.params[0]
    #dfull = dfull.loc[dfull.year_new.isin(pd.unique(dfgpp0.year_new))].copy()
    #dfull["dayfrac"] = (1-dfull.NIGHT)
    #dfull["seas_rain_mean5"] = np.mean(ymeans)
    #dfull["seas_rain_max5"] = np.mean(ymaxes)
    dfull["seas_rain_mean0"] = np.mean(ymeans0)
    dfull["seas_rain_max0"] = np.mean(ymaxes0)
    #%%
    btab = pd.DataFrame({"SITE_ID":site_id,
        "ddi":np.concatenate(ddlabel),
                         "G":np.concatenate(grec),
                         "ET":np.concatenate(et_plain),
                         "et_per_F_dm":et_topred,
                         "row0":row0,
                         "F":np.concatenate(frec),
                         "VPD":np.concatenate(vpd_plain),
                         "etcum":np.concatenate(etcum),
                         "smc":np.concatenate(smclist),
                         "doy":np.concatenate(dd_doy),
                         "date":np.concatenate(dd_dates),
                         "year":np.concatenate(ddyears),
                         "ddslopes":np.concatenate(individual_slopes),
                         "ddlen":np.concatenate([[len(x)]*len(x) for x in vpd_plain]),
                         "day_of_dd":np.concatenate([np.arange(len(x)) for x in vpd_plain])})


    #%%
    btab["cond"] = btab.ET/btab.VPD
    tau = -2/r1.params[0]
#%%

    etnormB = btab.ET**2/btab.F
    etnormB[np.abs(etnormB-np.mean(etnormB)) > 3*np.std(etnormB)] = np.nan

    btab["etnorm"] = etnormB
    btab["et2"] = btab.ET**2
    #btab["F2"] = btab.F**2
    #%%
    
    #%%
#    dmod = smf.ols("etnorm ~ 0 + etcum + C(ddi)",data=btab[btab.etcum > 0],missing='drop').fit()
    dmod = smf.ols("etnorm ~ 0 + etcum + C(ddi)",data=btab,missing='drop').fit()

    #dmod2 = smf.ols("et2 ~ 0 + etcum:F + C(ddi):F",data=btab,missing='drop').fit()

    dfull["tau_ddreg"] = -2/dmod.params.iloc[-1]
    dfull["tau_ddreg_hi"] = -2/(dmod.params.iloc[-1]+2*dmod.bse.iloc[-1])
    dfull["tau_ddreg_lo"] = -2/(dmod.params.iloc[-1]-2*dmod.bse.iloc[-1])
    dfull["tau_rel_err"] = -dmod.bse.iloc[-1]/dmod.params.iloc[-1]

    
    dmod2 = smf.ols("et2 ~ 0 + etcum:F + C(ddi):F",data=btab,missing='drop').fit()
    dfull["tau_ddreg2"] = -2/dmod2.params.iloc[0]
    dfull["tau_ddreg2_hi"] = -2/(dmod.params.iloc[-1]+2*dmod.bse.iloc[-1])
    dfull["tau_ddreg2_lo"] = -2/(dmod.params.iloc[-1]-2*dmod.bse.iloc[-1])
    dfull["tauET2_rel_err"] = -dmod2.bse.iloc[0]/dmod2.params.iloc[0]
#%%
    srec = dmod.predict(btab)/-dmod.params.iloc[-1]
    dfull["cor_retrieved_smc"] = cor_skipna(srec,btab.smc)[0]
    dfull["cor_retrieved_smc_pval"] = cor_skipna(srec,btab.smc)[1]
    btab["srec"] = srec
    #bmean = btab.groupby("ddi").mean(numeric_only=True)
    
    bfirst = btab.groupby("ddi").first()

    dfull["cor_retrieved_smc0"] = cor_skipna(bfirst.srec,bfirst.smc)[0]
    dfull["cor_retrieved_smc0_pval"] = cor_skipna(bfirst.srec,bfirst.smc)[1]
    #%%
    
    bfirst = btab.groupby("ddi").mean(numeric_only=True)

    dfull["cor_retrieved_smc0_ddmean"] = cor_skipna(bfirst.srec,bfirst.smc)[0]
    dfull["cor_retrieved_smc0_pval_ddmean"] = cor_skipna(bfirst.srec,bfirst.smc)[1]
#%%   
    ddlist.append(btab)
    
    #%%
    
    #tab1dd = tab1.groupby("ddi").mean(numeric_only=True).reset_index()
    tab1first = btab.groupby("ddi").first().reset_index()
    
    tab1first["et_init"] = 1*tab1first.ET
    tab1first["g_init"] = 1*tab1first.cond
    tab1first["s_init"] = tab1first.ET**2/2*tau/tab1first.F
#%%
    tab2 = pd.merge(btab,tab1first[["ddi","et_init","g_init","s_init"]],how="left",on="ddi")
    
    epredN = np.sqrt(btab.F*np.clip(dmod.predict(btab),0,np.inf))
    btab["etpred"] = epredN
    dfull["etr2_norm"] = r2_skipna(epredN/tab2.et_init,tab2.ET/tab2.et_init)
    dfull["gr2_norm"] = r2_skipna(epredN/tab2.VPD/tab2.g_init,tab2.cond/tab2.g_init)
    
    dfull["etr2_raw"] = r2_skipna(epredN,tab2.ET)
    dfull["gr2_raw"] = r2_skipna(epredN/tab2.VPD,tab2.cond)
    #%%
    all_results.append(dfull)
    
    site_message.append("Tau estimated")
    #%%
    #site_dd_tab = pd.DataFrame({"smc0":smc_start,"wl":is_limited,"smcavg":smc_avg,"smcend":smc_end}) #,
                                #"etavg":et_avg,"etinit":et_init,"lai":lai_avg,"vpd":vpd_avg})
    site_dd_tab = pd.DataFrame({"wl":is_limited,"smcavg":smc_avg}) #,

    site_dd_tab["SITE_ID"] = site_id
    site_dd_limit.append(site_dd_tab)
    
#%%
all_results = pd.concat(all_results)
all_results0 = pd.concat(all_results0)

#%%
site_dd_limit = pd.concat(site_dd_limit)
#%%
site_count = np.array(all_results.groupby("SITE_ID").count()["year"])
site_year = np.array(all_results.groupby("SITE_ID").nunique()["year"])

#%%
df1 = all_results.groupby("SITE_ID").first().reset_index()
df0 = all_results0.groupby("SITE_ID").first().reset_index()

#%%
def qt_gt1(x,q):
    return np.quantile(x[x >= 1],q)
def mean_gt1(x):
    return np.mean(x[x >= 1])
#%%
df1b = df1.loc[df1.gppR2_exp > 0].copy()
#df1b = df1.loc[df1.gppR2_hourly > 0].copy()

#%%
df_meta= df1b.loc[df1b.tau_ddreg > 0]

#%%
df_meta = pd.merge(df_meta,metadata,left_on="SITE_ID",right_on="fluxnetid",how="left")
#%%

df_meta = df_meta.loc[df_meta.tau_ddreg_lo > 0]
df_meta = df_meta.loc[df_meta.tau_ddreg_hi > 0]
#%%
df_meta = df_meta.loc[np.abs(df_meta.tau_ddreg_hi-df_meta.tau_ddreg_lo) < 100]

#%%
df_meta = df_meta.loc[df_meta.reg_ndd >= 3].copy()

#%%
df_meta["ddrain_mean"] = 1*df_meta.seas_rain_max0
df_meta["ddrain_2mean"] = 1*df_meta.seas_rain_mean0
#%%
df_meta["gsrain_len"] = df_meta.gslen_annual
#%%
#df_meta = df_meta.loc[df_meta.frac_gt9 > 0.01].copy()
#df_meta = df_meta.loc[df_meta.frac_gt9 < 0.99].copy()

df_meta = df_meta.loc[df_meta.gpp_fit_frac_above9 > 0.05].copy()
df_meta = df_meta.loc[df_meta.gpp_fit_frac_above9 < 0.95].copy()
#df_meta = df_meta.loc[df_meta.max_resid_cor < 0.5].copy()
#%%
rainmod = smf.ols("tau_ddreg ~ ddrain_mean",data=df_meta).fit()
#%%
r2_11 = 1-np.mean((df_meta.ddrain_mean-df_meta.tau_ddreg)**2)/np.var(df_meta.tau_ddreg)
print(r2_11)

fig,ax = plt.subplots(1,1,figsize=(10,8))

lmax = 1.1*np.max(df_meta.ddrain_mean)

betas = np.array(np.round(np.abs(rainmod.params),2)).astype(str)
if rainmod.params[0] < 0:
    reg_eqn = r"$\tau$ = "+betas[1]+"$D_{max}$"+" - "+betas[0]
else:
    reg_eqn = r"$\tau$ = "+betas[1]+"$D_{max}$"+" + "+betas[0]
r2_txt = "($R^2$ = " + str(np.round(rainmod.rsquared,2)) + ")"
reg_lab = "Regression line" + "\n" + reg_eqn + "\n" + r2_txt
line2, = ax.plot([0,lmax],np.array([0,lmax])*rainmod.params[1]+rainmod.params[0],"b",label=reg_lab)
line1, = ax.plot([0,lmax],[0,lmax],"k--",label="1:1 line")

#plt.plot([0,150],np.array([0,150])*reg0.params[0],"b--",label="Regression line\n($R^2$ = 0.39)")
#leg1 = ax.legend(loc="upper left")
#leg1 = ax.legend(loc="lower right")
leg1 = ax.legend()

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.ddrain_mean,subI.tau_ddreg,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
xmax = np.max(df_meta.ddrain_mean)
ymax = np.max(df_meta.tau_ddreg)


ax.set_xlim(0,1.1*xmax)
ax.set_ylim(0,1.1*ymax)
ax.set_xlabel("Growing season $D_{max}$ (days)",fontsize=24)
ax.set_ylabel(r"$\tau$ (days)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=3 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")
#%%
import cartopy.crs as ccrs
import cartopy.feature as cf
#%%
fig = plt.figure(figsize=(15,15),dpi=100)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.stock_img()
ax.add_feature(cf.LAKES)
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
ax.plot(df_meta.LOCATION_LONG,df_meta.LOCATION_LAT,'*',alpha=0.75,color="red",markersize=10,markeredgecolor="gray")
ax.set_xlim(np.min(df_meta.LOCATION_LONG)-7,np.max(df_meta.LOCATION_LONG)+7)
ax.set_ylim(np.min(df_meta.LOCATION_LAT)-7,np.max(df_meta.LOCATION_LAT)+7)
#%%

#%%
fig,ax = plt.subplots(1,1,figsize=(10,8))

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta.loc[df_meta.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.mat_data,subI.map_data*365/10,'o',alpha=0.75,markersize=15,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,210)
#ax.set_ylim(0,210)
ax.set_xlabel("Average temperature ($^oC)$",fontsize=24)
ax.set_ylabel("Average annual precip. (cm)",fontsize=24)

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=3 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%
df_meta3 = df_meta.sort_values("etr2_norm")
df_meta3["et_rank"] = np.arange(len(df_meta3))

fig,axes = plt.subplots(3,1,figsize=(14,12))
ax = axes[2]

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.et_rank,subI.etr2_norm,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
ax.set_xticks(df_meta3.et_rank,df_meta3.SITE_ID,rotation=90,fontsize=14)
#ax.set_xlim(0,250)
ax.set_ylim(0,1)
#ax.set_xlabel("Rank",fontsize=24)
ax.set_title(r"$R^2$ of $ET/ET_{0}$ during water-limited drydowns",fontsize=24)

#df_meta["r2_retrieved_smc"] = df_meta["cor_retrieved_smc"]**2
df_meta3 = df_meta.sort_values("cor_retrieved_smc")
df_meta3["s_rank"] = np.arange(len(df_meta3))
ax = axes[1]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.s_rank,subI.cor_retrieved_smc,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.s_rank,df_meta3.SITE_ID,rotation=90,fontsize=14)
ax.set_title(r"Corr. of modeled and observed soil moisture during water-limited drydowns",fontsize=24)
#ax.axhline(0,color='k')

df_meta3 = df_meta.sort_values("gppR2_exp")
df_meta3["gpp_rank"] = np.arange(len(df_meta3))
ax = axes[0]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.gpp_rank,subI.gppR2_exp,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.gpp_rank,df_meta3.SITE_ID,rotation=90,fontsize=14)
ax.set_title(r"$R^2$ of GPP given observed g during growing season",fontsize=24)
fig.tight_layout()
fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.02),ncols=3)
#%%
biome_index = dict(zip(biome_list,range(len(biome_list))))
df_meta["biome_number"] = [biome_index[x] for x in df_meta.combined_biome]
#%%
plot_colors = mpl.colormaps["tab10"](df_meta["biome_number"] +2)
df_meta["tau"] = 1*df_meta.tau_ddreg
#%%
def myplot(ax,x,y,xlab,ylab):
    ax.scatter(x,y,c=plot_colors)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    myr2 = np.corrcoef(x,y)[0,1]**2
    ax.text(0.1,0.85,  "$R^2$ = " + str(np.round(myr2,2)),transform=ax.transAxes)
#%%
fig,axes=plt.subplots(3,3,figsize=(12,10))

myplot(axes[2,0],df_meta.Aridity,df_meta.tau,
       "Annual AI",r"$\tau$ (days)")
myplot(axes[2,1],df_meta.Aridity_gs,df_meta.tau,
       "GrowSeas AI","")
axes[2,2].set_axis_off()
# myplot(axes[2,2],df_meta.ddrain_mean,df_meta.tau,
#        "GrowSeas $D_{max}$","")
# axes[2,2].plot([0,100],[0,100],'k--',label="1:1 line")
# axes[2,2].legend(loc="lower right",fontsize=14)

myplot(axes[1,0],df_meta.map_data,df_meta.tau,
       "Annual P (mm/day)",r"$\tau$ (days)")

myplot(axes[1,1],df_meta.gsrain_mean,df_meta.tau,
       "GrowSeas P (mm/day)","")

myplot(axes[1,2],df_meta.gsrain_len,df_meta.tau,
       "GrowSeas length (days)","")

myplot(axes[0,0],df_meta.fullyear_dmean,df_meta.tau,
       "Annual $D_{mean}$ (days)",r"$\tau$ (days)")
myplot(axes[0,1],df_meta.fullyear_dmax,df_meta.tau,
       "Annual $D_{max}$ (days)","")
myplot(axes[0,2],df_meta.ddrain_2mean,df_meta.tau,
       "GrowSeas $D_{mean}$ (days)","")
axes[0,2].plot([0,np.max(df_meta.ddrain_2mean)],[0,np.max(df_meta.ddrain_2mean)],'k--',label="1:1 line")
axes[0,2].legend(loc="lower right",fontsize=14)

fig.tight_layout()

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=3 )
#%%
def in_bounds(x):
    return (x < np.sort(x)[-2]) #*(x > np.sort(x)[1])
inliers = np.ones(len(df_meta))
for col in ["tau","Aridity","Aridity_gs","ddrain_mean","map_data","gsrain_mean","gsrain_len","fullyear_dmean","fullyear_dmax","ddrain_2mean"]:
    inliers *= np.array(in_bounds(df_meta[col]))
inliers = inliers==1
# #%%
def myplot2(ax,x0,y0,xlab,ylab):
    x = x0[inliers]
    y = y0[inliers]

    ax.scatter(x,y,c=plot_colors[inliers])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(0,70)
    myr2 = np.corrcoef(x,y)[0,1]**2
    ax.text(0.1,0.85,  "$R^2$ = " + str(np.round(myr2,2)),transform=ax.transAxes)
#%%
fig,axes=plt.subplots(3,3,figsize=(12,10))

myplot2(axes[2,0],df_meta.Aridity,df_meta.tau,
       "Annual AI",r"$\tau$ (days)")
myplot2(axes[2,1],df_meta.Aridity_gs,df_meta.tau,
       "GrowSeas AI","")
#axes[2,2].set_axis_off()
myplot2(axes[2,2],df_meta.ddrain_mean,df_meta.tau,
       "GrowSeas $D_{max}$","")
axes[2,2].plot([0,50],[0,50],'k--',label="1:1 line")
axes[2,2].legend(loc="lower right",fontsize=14)
myplot2(axes[1,0],df_meta.map_data,df_meta.tau,
       "Annual P (mm/day)",r"$\tau$ (days)")

myplot2(axes[1,1],df_meta.gsrain_mean,df_meta.tau,
       "GrowSeas P (mm/day)","")

myplot2(axes[1,2],df_meta.gsrain_len,df_meta.tau,
       "GrowSeas length (days)","")

myplot2(axes[0,0],df_meta.fullyear_dmean,df_meta.tau,
       "Annual $D_{mean}$ (days)",r"$\tau$ (days)")
myplot2(axes[0,1],df_meta.fullyear_dmax,df_meta.tau,
       "Annual $D_{max}$ (days)","")
myplot2(axes[0,2],df_meta.ddrain_2mean,df_meta.tau,
       "GrowSeas $D_{mean}$ (days)","")
axes[0,2].plot([0,50],[0,50],'k--',label="1:1 line")
axes[0,2].legend(loc="lower right",fontsize=14)

fig.tight_layout()

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=3 )
#%%
plt.figure(figsize=(7,7))
#plt.plot(df_meta.summer_end-df_meta.summer_start,df_meta.ddrain_mean,'o')
plt.plot(df_meta.gsrain_len,df_meta.ddrain_mean,'o')

plt.xlabel("GS length (days)",fontsize=22)
plt.ylabel("$D_{max}$ (days)",fontsize=22)
#%%

#%%
def nanterp(x):
    xind = np.arange(len(x))
    return np.interp(xind,
                      xind[np.isfinite(x)],
                      x[np.isfinite(x)])
#%%
ddlist = pd.concat(ddlist)
#%%
#tabS = ddlist.loc[ddlist.SITE_ID=="US-Me5"]
site_pair = ["US-Me5","US-SRM"]
plt.figure()
si = 1
for x in site_pair:
    #plt.subplot(2,1,si)
    tabS = ddlist.loc[ddlist.SITE_ID==x].copy()
    tabS = tabS.loc[tabS.year >= 2001].copy()
    tabS = tabS.loc[tabS.ddlen >= 10].copy()
    #ddlens = tabS.groupby("ddi").count().reset_index()
    #longDD = np.argmax(ddlens.et_per_F_dm)
    #longDD = np.argmin(np.abs(ddlens.et_per_F_dm-18))
    ddfirst = tabS.groupby("ddi").first().reset_index()
    #longDD = np.argmax(ddlens.et_per_F_dm)
    #longDD = np.argmin(np.abs(ddlens.et_per_F_dm-18))
    if x=="US-Me5":
        longDD = ddfirst.loc[np.abs(ddfirst.ET-1) < 0.1].ddi.iloc[0]
        #np.argmin(np.abs(ddfirst.ET-1))
    if x=="US-SRM":
        longDD = ddfirst.loc[np.abs(ddfirst.ET-2.2) < 0.1].ddi.iloc[0]

        #longDD = np.argmin(np.abs(ddfirst.ET-2.2))
    jtab = tabS[tabS.ddi==longDD].reset_index()
    istart = np.where(np.isfinite(jtab.ET))[0][0]
    jtab = jtab.iloc[istart:].copy()
    tau = 20
    #sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    #epred20 = np.sqrt(2/tau*jtab.F2*(sm_init-jtab.etcum))
    term1 = -1/tau*np.sqrt(jtab.F)*jtab.G
    #c2 = (np.nanmean(jtab.ET) - np.nanmean(term1))/np.nanmean(jtab.F)
    c2 = jtab.ET.iloc[0]/np.sqrt(jtab.F.iloc[0])
    # sm0 = 10
    # c1 = np.sqrt(sm0*4)
    # c2 = 0.5*c1*np.sqrt(2/tau)
    epred20 = np.clip(term1 + c2*np.sqrt(jtab.F),0,np.inf)
    
    tau = 50
    term1 = -1/tau*jtab.F*jtab.G
    
    c2 = jtab.ET.iloc[0]/np.sqrt(jtab.F.iloc[0])
    
    epred50 =  np.clip(term1 + c2*np.sqrt(jtab.F),0,np.inf)
    
    xvar = np.arange(len(jtab.ET))+2
    plt.plot(xvar, np.array(jtab.ET),'ko--',linewidth=3,label="Eddy covariance")
    plt.plot(xvar,epred50,'o-',color="tab:blue",linewidth=3,alpha=0.6,label=r"Model, $\tau$ = 50 days")
    plt.plot(xvar,epred20,'o-',color="tab:orange",linewidth=3,alpha=0.6,label=r"Model, $\tau$ = 20 days")
    if si == 1:
        plt.legend(fontsize=16)
    si += 1
#plt.tight_layout()
#plt.ylim(-0.1,3.9)
plt.xticks(np.arange(2,23,3))
plt.xlabel("Day of drydown",fontsize=22)
plt.ylabel("ET (mm/day)",fontsize=22)
plt.text(1.5,1.25,site_pair[0],fontsize=20)
plt.text(1.5,2.7,site_pair[1],fontsize=20)
plt.ylim(-0.1,3)
#%%
from statsmodels.stats.anova import anova_lm

biome_diff = anova_lm(smf.ols("tau_ddreg ~ C(combined_biome)",data=df_meta).fit())
#%%
site_year2 = ddlist.groupby("SITE_ID").nunique().reset_index()
site_year2["N_year"] = site_year2.year
df_export = df_meta[["SITE_ID",'SITE_NAME',"combined_biome",'koeppen_climate',
'LOCATION_LAT','LOCATION_LONG','LOCATION_ELEV',
'gsrain_len','map_data','gsrain_mean','mat_data',
'Aridity','Aridity_gs',
'fullyear_prain','gs_prain',
'fullyear_dmax','fullyear_dmean','seas_rain_max0','seas_rain_mean0',
'gppR2_exp','reg_ndd','reg_npoints',
'tau_ddreg','tau_ddreg_lo','tau_ddreg_hi',
'etr2_norm','gr2_norm',"cor_retrieved_smc","tau_rel_err"]].copy()
df_export = pd.merge(df_export,site_year2[["SITE_ID","N_year"]],on="SITE_ID",how="left")
newcolnames = "SITE_ID	SITE_NAME	Biome	koeppen_climate	LOCATION_LAT	LOCATION_LONG	LOCATION_ELEV	GrowSeas_length_days	MeanPrec_Annual_mmday	MeanPrec_GrowSeas_mmday	MeanAnnualTemp_degC	AridityIndex_annual	AridityIndex_GrowSeas	RainFreq_annual_perday	RainFreq_GrowSeas_perday	Dmax_annual_days	Dmean_annual_days	Dmax_GrowSeas_days	Dmean_GrowSeas_days	GPP_model_R2	N_drydowns_used	Total_drydown_days_used	Tau_days	Tau_95ci_low_days	Tau_95ci_high_days	ET_norm_predict_R2	g_norm_predict_R2   soil_mois_model_cor    a_slope_relative_standard_error 	N_years_of_data".split()
df_export.columns = newcolnames
#%%
xt = df_meta.tau
print("Tau, lower upper mean: ")
print(np.min(xt),np.max(xt),np.mean(xt))
xt = df_meta.tau_ddreg_hi - df_meta.tau_ddreg_lo
print("Tau 95% CI, lower upper mean: ")
print(np.min(xt),np.max(xt),np.mean(xt))
xt = df_meta.reg_ndd
print("N dd, lower upper mean: ")
print(np.min(xt),np.max(xt),np.mean(xt))
xt = df_export.N_years_of_data
print("N year, lower upper mean: ")
print(np.min(xt),np.max(xt),np.mean(xt))

xt = df_meta.gppR2_exp
print("GPP R2, lower upper mean: ")
print(np.quantile(xt,0.25),np.quantile(xt,0.75),np.mean(xt))


xt = df_meta.etr2_norm
print("ET R2, lower upper mean positive: ")
print(np.quantile(xt,0.25),np.quantile(xt,0.75),np.mean(xt), np.sum(xt>0))

xt = df_meta.gr2_norm
print("g R2, lower upper mean: ")
print(np.quantile(xt,0.25),np.quantile(xt,0.75),np.mean(xt) , np.sum(xt>0))

#%%
xt = df_meta.cor_retrieved_smc
print("g R2, lower upper mean: ")
print(np.nanquantile(xt,0.25),np.nanquantile(xt,0.75),np.nanmean(xt) )
#%%
xt = df_meta.mat_data
print("Annual temp, lower upper: ")
print(np.min(xt),np.max(xt))

xt = df_meta.map_data*365/10
print("Annual precip, lower upper: ")
print(np.min(xt),np.max(xt))
#%%
site_meanLAI = all_results.groupby("SITE_ID").mean(numeric_only=True).reset_index()[["SITE_ID","LAI"]]
df_meta2 = pd.merge(df_meta,site_meanLAI,on='SITE_ID',how="left")
laimod = smf.ols("etr2_norm ~ LAI_y", data=df_meta2.loc[df_meta2.etr2_norm > 0]).fit()
#%%
fig, ax = plt.subplots(2,2,figsize=(10,8))
ax[0,0].plot(df_meta.reg_ndd, df_meta.tau_rel_err,'o'); 
ax[0,0].set_xlabel("Number of drydowns analyzed")

ax[0,1].plot(df_meta.reg_npoints, df_meta.tau_rel_err,'o'); 
ax[0,1].set_xlabel("Number of days analyzed")

ax[1,0].plot(df_meta.gppR2_exp, df_meta.tau_rel_err,'o'); 
ax[1,0].set_xlabel("$R^2$ of GPP model")

ax[1,1].set_axis_off()

fig.supylabel("Relative standard error of regression slope")
fig.tight_layout()
