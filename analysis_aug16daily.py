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
#import glob
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
    return scipy.stats.pearsonr(x[goodxy],y[goodxy])

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

bigyear = pd.read_csv("data_with_daytime_aug21.csv")

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

#%%
all_results = []
ddlist = []

for site_id in pd.unique(bigyear.SITE_ID):
    #%%
    #if site_id=="ZM-Mon":
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
    dfday = dfull[["par","LAI","cond2","gpp","vpd","rain","rain_prev"]].dropna()
    dfday = dfday.loc[dfday.rain==0].copy()
    dfday = dfday.loc[dfday.rain_prev==0].copy()

    #%%
    use_mm = 1
    if use_mm:
        def tofit(pars):
            amax1,kA,gmax1,kG = pars
            
            amax = amax1*dfday.par/(dfday.par + kA) * dfday.LAI
            gA = gmax1*dfday.par/(dfday.par + kG) * dfday.LAI
    
            gpp_pred = amax*(1-np.exp(-dfday.cond2/gA))
            z = (gpp_pred-dfday.gpp)#[dfday.VPD > 1]
            return z
        himean = np.quantile(dfday.gpp/dfday.LAI,0.9)
        fit0 = np.array([himean,600,himean/200,600])
        myfit = scipy.optimize.least_squares(tofit,x0=fit0,method="lm",x_scale=np.abs(fit0))

        amax1,kA,gmax1,kG = myfit.x
    #%%
    dfull["amax1"] = amax1
    dfull["kA1"] = kA
    dfull["gmax1"] = gmax1
    dfull["kG1"] = kG
    
    # #%%
    dfull["amax_d2"] = amax1*dfull.par/(dfull.par + kA) * dfull.LAI
    dfull["gA_d2"] = gmax1*dfull.par/(dfull.par + kG) * dfull.LAI

    dfull["gpp_pred_d2"] = dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2))
    
    dfull["kgpp"] = dfull.gA_d2#*dfull.dayfrac
    
    dfull["gpp_pred_d2a"] = 1.5*dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/1.5))
    dfull["gpp_pred_d2b"] = 0.5*dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/0.5))
    
    
    dfull["gpp_pred_d2c"] = dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/2))
    dfull["gpp_pred_d2d"] = dfull.amax_d2*(1-np.exp(-dfull.cond2/dfull.gA_d2/0.5))
    
   
    #%%
    gAtest = gmax1*dfday.par/(dfday.par + kG) * dfday.LAI
    z1 = 1-np.exp(-dfday.cond2/gAtest)
    dfull["frac_gt9"] = np.mean(z1 > 0.9)
    
    residG = np.log(dfull.gpp/dfull.gpp_pred_d2)
    residcors = []
    for var1 in ["airt","doy","LAI","par","smc"]:
        try:
            residcors.append(cor_skipna(dfull[var1],residG)[0])
        except:
            pass
    dfull["max_resid_cor"] = np.max(np.abs(np.array(residcors)))
    
    #%%
    dfull["gppR2_exp"] = r2_skipna(dfull.gpp_pred_d2,dfull.gpp)
    if r2_skipna(dfull.gpp_pred_d2,dfull.gpp) < 0:
        site_message.append("GPP model did not fit")
        continue
    
    #smin_mm = -500
    #tauDay = 50
    #%%
    dfGS = dfull.loc[dfull.is_summer].copy()
    dfull["gsrain_mean"] = np.mean(dfGS.rain)
#    dfGS = dfull.copy()
    #dfGS["cond_per_LAI"] = dfGS.cond/dfGS.LAI
    #cl75 =  np.nanquantile(dfGS["cond_per_LAI"],0.75)
#%%
    seaslens = []
    ddreg_fixed = []
    #ddreg_random = []
    et_over_dd = []
    ymaxes = []
    ymeans= []
    
    ymaxes0 = []
    ymeans0 = []
    # ddreg_fixed2 = []
    # et_over_dd2 = []
    
    ddlabel = []
    ddii = 0
    
    grec = []
    frec = []
    et_plain = []
    vpd_plain = []
    etcum = []
    ddyears = []
    
    ddall = 0
    
    #%%
    for y0 in pd.unique(dfGS.year_new):
    #%%
        dfy = dfGS.loc[dfGS.year_new==y0].copy()
        if np.sum(np.isfinite(dfy.ET)) < 10:
            continue
        #mederr = np.nanmedian(dfy.gpp / dfy.gpp_pred)
        #dfy.kgpp = 1.8/45*dfy.LAI
        #dfy["gmax"] = 4*dfy.LAI
        #dfy.kgpp *= 1.3
        #doy_arr = np.arange(dfy.doy.iloc[0],dfy.doy.iloc[-1]+1)
        doy_indata = np.array(dfy.doy)
        vpd_arr = np.clip(np.array(dfy.vpd),0.1,np.inf)/100
        if np.sum(np.isfinite(vpd_arr)) < 25:
            continue
        vpd_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(vpd_arr)],
                            vpd_arr[np.isfinite(vpd_arr)])
        k_mm_day = np.array(dfy.kgpp)*mol_s_to_mm_day #* np.array(dfy.gpp/dfy.gpp_pred)
        rain_arr = np.array(dfy.rain)
        seaslens.append(len(rain_arr))
        #%%
    #et_out = petVnum_samp*final_cond/(gammaV*(fac1 + final_cond)+sV_samp*final_cond)/44200
        
        et_mmday = np.array(dfy.ET)*mol_s_to_mm_day
        
        #et_mmday = np.array(dfy.cond)*mol_s_to_mm_day
        
        et_mmday_interp = np.interp(doy_indata,
                            doy_indata[np.isfinite(et_mmday)],
                            et_mmday[np.isfinite(et_mmday)])
        # w_arr = np.array(dfy.waterbal)
        
        #cond1 = np.array(dfy.cond_per_LAI)
        #et_mmday[cond1 > cl75] = np.nan
        
        #et_mmday[dfy.airt < 10] = np.nan
        et_mmday[dfy.par < 100] = np.nan

        et_mmday[dfy.vpd <= 0.5] = np.nan
        
        #et_mmday[dfy.ET_qc < 0.5] = np.nan
        
#%%
        rain_d5 = np.array([0] + list(np.where(rain_arr > 5)[0]) + [len(rain_arr)])
        dd5= np.diff(rain_d5)
        ymaxes.append(np.max(dd5))
#        ymeans.append(np.mean(dd5[dd5 >= 2]))
        ymeans.append(np.sum(dd5**2)/np.sum(dd5))

#%%
        rain_days = np.array([0] + list(np.where(rain_arr > 0)[0]) + [len(rain_arr)])
        ddgood = np.where(np.diff(rain_days) >= 7)[0]
        
        ddall += len(ddgood)
        
        ddstart = rain_days[ddgood]+2
        ddend = rain_days[ddgood+1]
    
        etnorm = et_mmday**2 / (vpd_interp*k_mm_day)
        #etnorm[vpd_arr < 0.1/100] = np.nan

        
        #etnorm[dfy.airt < 10] = np.nan
        #etnorm[dfy.par < 100] = np.nan

        #etnorm[dfy.vpd < 0.5] = np.nan
        #etnorm[dfy.ET_qc < 0.5] = np.nan
        
        doyY = np.array(dfy.doy)
    #%%
        rain_days = np.array([0] + list(np.where(rain_arr > 0)[0]) + [len(rain_arr)])

        dd0= np.diff(rain_days) #- 1
        ymaxes0.append(np.max(dd0))
        #ymeans0.append(np.mean(dd0[dd0 >= 2]))
        ymeans0.append(np.sum(dd0**2)/np.sum(dd0))

    #%%
        #tau_with_unc = []
        #winit_with_unc = []
        for ddi in range(len(ddstart)):
            starti = ddstart[ddi]
            #starti = max(ddstart[ddi],ddend[ddi] - 50) 

            endi = ddend[ddi]
            #endi = min(starti+20,ddend[ddi])
            f_of_t = (vpd_arr*k_mm_day)[starti:endi]
#           # g_of_t = np.cumsum(np.sqrt(f_of_t))
#            g_of_t = np.array([0] + list(np.cumsum(np.sqrt(f_of_t))))[:-1]
            g_of_t = np.array([0] + list(np.cumsum(np.sqrt(f_of_t))))[:-1]
            #g_of_t = g_of_t[:20]
            #yfull = et_mmday[ddstart[ddi]:ddend[ddi]]/np.sqrt(f_of_t)
            #yfull = yfull[:20]
            
            doyDD = doyY[starti:endi]
            yfull = etnorm[starti:endi]#[:20]
            etsel = et_mmday_interp[starti:endi]#[:20]
            rainsel =  rain_arr[starti:endi]
            #if r1.params[1] < 0 and r1.pvalues[1] < 0.05:
            if np.sum(np.isfinite(yfull)) >= 5: # and np.mean(np.isfinite(yfull)) >= 0.75:
                #et_over_dd.append(yfull - np.nanmean(yfull))
                #ddreg_fixed.append(g_of_t - np.mean(g_of_t[np.isfinite(yfull)]))
                etcumDD = np.array([0] + list(np.cumsum(etsel-rainsel)))[:-1]

                rDD = sm.OLS(yfull,sm.add_constant(etcumDD),missing='drop').fit()
#                rDD = sm.OLS(yfull,sm.add_constant(g_of_t),missing='drop').fit()

#                if rDD.pvalues[1] < 0.1 and rDD.params[1] < 0:
                if rDD.rsquared > 0.25 and rDD.params[1] < 0:
                                       
                    
                    ddlabel.append([ddii]*len(yfull))
                    ddyears.append([y0]*len(yfull))
    
                    
                    frec.append(f_of_t)
                    grec.append(g_of_t)
                    vpd_plain.append(vpd_arr[starti:endi])
                    et_plain.append(et_mmday[starti:endi])
                    
                    etcum.append(etcumDD)
                    
                    et_over_dd.append(yfull - np.nanmean(yfull))
                    ddreg_fixed.append(etcumDD - np.mean(etcumDD[np.isfinite(yfull)]))
                    
                    #et_over_dd.append((yfull - np.nanmean(yfull))/np.std(etcumDD))
                    #ddreg_fixed.append((etcumDD - np.mean(etcumDD[np.isfinite(yfull)]))/np.std(etcumDD))
    
                    ddii += 1
        #%%
        
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
    #%%
    dfull["tau_ddreg"] = -2/r1.params[0]
    dfull["tau_ddreg_lo"] = -2/(r1.params[0] - 2*r1.bse[0])
    dfull["tau_ddreg_hi"] = -2/(r1.params[0] + 2*r1.bse[0])
    #%%
    dfull["gslen_annual"] = np.mean(seaslens)
    dfull["tau_rel_err"] = -r1.bse[0]/r1.params[0]
    #dfull = dfull.loc[dfull.year_new.isin(pd.unique(dfgpp0.year_new))].copy()
    #dfull["dayfrac"] = (1-dfull.NIGHT)
    dfull["seas_rain_mean5"] = np.mean(ymeans)
    dfull["seas_rain_max5"] = np.mean(ymaxes)
    dfull["seas_rain_mean0"] = np.mean(ymeans0)
    dfull["seas_rain_max0"] = np.mean(ymaxes0)
    #%%
    btab = pd.DataFrame({"SITE_ID":site_id,
        "ddi":np.concatenate(ddlabel),
                         "G":np.concatenate(grec),
                         "ET":np.concatenate(et_plain),
                         "et_per_F_dm":et_topred,
                         "row0":row0,
                         "F":np.sqrt(np.concatenate(frec)),
                         "VPD":np.concatenate(vpd_plain),
                         "etcum":np.concatenate(etcum),
                         "year":np.concatenate(ddyears),
                         "ddlen":np.concatenate([[len(x)]*len(x) for x in vpd_plain])})
    #%%
    btab["cond"] = btab.ET/btab.VPD
    tau = -2/r1.params[0]
#%%
    btab["etnorm"] = btab.ET**2/btab.F**2
    btab["et2"] = btab.ET**2
    btab["F2"] = btab.F**2
    #%%
    
    #%%
    dmod = smf.ols("et2 ~ 0 + etcum:F2 + C(ddi):F2",data=btab,missing='drop').fit()
    dmod0 = smf.ols("et2 ~ 0 + C(ddi):F2",data=btab,missing='drop').fit()
    dmod2 = smf.ols("et2 ~ 0 + etcum:F2 + np.power(etcum,2):F2 + C(ddi):F2",data=btab,missing='drop').fit()

    #%%
    ddlist.append(btab)
    
    #%%
    
    #tab1dd = tab1.groupby("ddi").mean(numeric_only=True).reset_index()
    tab1first = btab.groupby("ddi").first().reset_index()
    
    tab1first["et_init"] = 1*tab1first.ET
    tab1first["g_init"] = 1*tab1first.cond

    tab2 = pd.merge(btab,tab1first[["ddi","et_init","g_init"]],how="left",on="ddi")


    tab2["mydiff"] = tab2.et2*tau/2 + tab2.etcum*tab2.F2
    dmod2 = smf.ols("mydiff ~ 0 + C(ddi):F2",data=tab2,missing='drop').fit()
    epredN = np.sqrt(np.clip(dmod2.predict(tab2)*2/tau - tab2.etcum*tab2.F2*2/tau,0,np.inf))

    #dmod = smf.ols("et2 ~ 0 + etcum:F2 + C(ddi):F2",data=tab2,missing='drop').fit()
    #epredM = np.sqrt(np.clip(dmod.predict(tab2),0,np.inf))
    btab["etpred"] = epredN
    dfull["etr2_norm"] = r2_skipna(epredN/tab2.et_init,tab2.ET/tab2.et_init)
    dfull["gr2_norm"] = r2_skipna(epredN/tab2.VPD/tab2.g_init,tab2.cond/tab2.g_init)
    dfull["tau_simult"] = -2/dmod.params[0]
    #%%
    all_results.append(dfull)
    
    site_message.append("Tau estimated")
    #%%
    # plt.figure(figsize=(14,10))
    # plt.subplot(2,3,1)
    # plt.plot(dfull.ET)
    # plt.title("ET")
    # plt.subplot(2,3,2)
    # plt.plot(dfull.vpd)
    # plt.title("VPD")
    # plt.subplot(2,3,3)
    # plt.plot(dfull.gpp)
    # plt.title("GPP")
    # plt.subplot(2,3,4)
    # plt.plot(dfull.gA_daily)
    # plt.title("gA")
    # plt.subplot(2,3,5)
    # plt.plot(dfull.cond)
    # plt.title("g")
    # plt.subplot(2,3,6)
    # plt.plot(dfull.LAI)
    # plt.title("LAI")
    # plt.suptitle(site_id)
    # #%%
    # plt.figure()
    # plt.plot(row0,et_topred,'o')
    # plt.title(site_id)
#%%
 
#%%
all_results = pd.concat(all_results)
#%%

#%%
site_count = np.array(all_results.groupby("SITE_ID").count()["year"])
site_year = np.array(all_results.groupby("SITE_ID").nunique()["year"])

#%%
df1 = all_results.groupby("SITE_ID").first().reset_index()

#%%
def qt_gt1(x,q):
    return np.quantile(x[x >= 1],q)
def mean_gt1(x):
    return np.mean(x[x >= 1])
#%%
df1b = df1.loc[df1.gppR2_exp > 0].copy()

#%%
#%%
df_meta= df1b.loc[df1b.tau_ddreg > 0]

#%%
df_meta = pd.merge(df_meta,metadata,left_on="SITE_ID",right_on="fluxnetid",how="left")
#%%

df_meta = df_meta.loc[df_meta.tau_ddreg_lo > 0]
df_meta = df_meta.loc[df_meta.tau_ddreg_hi > 0]
#%%

df_meta = df_meta.loc[df_meta.reg_ndd >= 3].copy()

#%%
df_meta["ddrain_mean"] = 1*df_meta.seas_rain_max0
df_meta["ddrain_2mean"] = 1*df_meta.seas_rain_mean0
#%%
df_meta["gsrain_len"] = df_meta.gslen_annual
#%%
df_meta = df_meta.loc[df_meta.frac_gt9 > 0.01].copy()
df_meta = df_meta.loc[df_meta.frac_gt9 < 0.99].copy()
#df_meta = df_meta.loc[df_meta.max_resid_cor < 0.5].copy()
#%%
rainmod = smf.ols("tau_ddreg ~ ddrain_mean",data=df_meta).fit()
#%%
r2_11 = 1-np.mean((df_meta.ddrain_mean-df_meta.tau_ddreg)**2)/np.var(df_meta.tau_ddreg)
print(r2_11)

fig,ax = plt.subplots(1,1,figsize=(10,8))

lmax = 1.1*np.max(df_meta.ddrain_mean)

#line1, = ax.plot([0,lmax],[0,lmax],"k",label="1:1 line, $R^2$=0.59")
betas = np.array(np.round(np.abs(rainmod.params),2)).astype(str)
if rainmod.params[0] < 0:
    reg_eqn = r"$\tau$ = "+betas[1]+"$D_{max}$"+" - "+betas[0]
else:
    reg_eqn = r"$\tau$ = "+betas[1]+"$D_{max}$"+" + "+betas[0]
r2_txt = "($R^2$ = " + str(np.round(rainmod.rsquared,2)) + ")"
reg_lab = "Regression line" + "\n" + reg_eqn + "\n" + r2_txt
line2, = ax.plot([0,lmax],np.array([0,lmax])*rainmod.params[1]+rainmod.params[0],"b--",label=reg_lab)
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

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

#ax.add_artist(leg1)

#plt.savefig("C:/Users/nholtzma/OneDrive - Stanford/agu 2022/plots for poster/rain_scatter4.svg",bbox_inches="tight")
#%%
#allrain = pd.read_csv("year_round_rain_stats.csv")
#allrain = pd.read_csv("gs_start_rain_stats_maxLAI.csv")
#df_meta = pd.merge(df_meta,allrain,on="SITE_ID",how='left')

#%%
# yscaler = np.sqrt(zsoil_mol)
# molm2_to_mm = 18/1000
# s2day = 60*60*24

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
#df_meta = pd.merge(df_meta,df_base[["SITE_ID","Aridity","Aridity_gs","mat_data","map_data"]],on="SITE_ID",how="left")


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

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%


#%%
df_meta3 = df_meta.sort_values("etr2_norm")
df_meta3["et_rank"] = np.arange(len(df_meta3))

fig,axes = plt.subplots(3,1,figsize=(16,10))
ax = axes[1]

points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.et_rank,subI.etr2_norm,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)
ax.set_xticks(df_meta3.et_rank,df_meta3.SITE_ID,rotation=90)
#ax.set_xlim(0,250)
ax.set_ylim(0,1)
#ax.set_xlabel("Rank",fontsize=24)
ax.set_title(r"$R^2$ of $ET/ET_{0}$ during water-limited drydowns",fontsize=24)

#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")

df_meta3 = df_meta.sort_values("gr2_norm")
df_meta3["g_rank"] = np.arange(len(df_meta3))
ax = axes[2]
points_handles = []
for i in range(len(biome_list)):
    subI = df_meta3.loc[df_meta3.combined_biome==biome_list[i]]
    if len(subI) > 0:
        pointI, = ax.plot(subI.g_rank,subI.gr2_norm,'o',alpha=0.75,markersize=10,color=mpl.colormaps["tab10"](i+2),label=biome_list[i])
        points_handles.append(pointI)

#ax.set_xlim(0,250)
ax.set_ylim(0,1)
ax.set_xticks(df_meta3.g_rank,df_meta3.SITE_ID,rotation=90)
ax.set_title(r"$R^2$ of $g/g_{0}$ during water-limited drydowns",fontsize=24)
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
ax.set_xticks(df_meta3.gpp_rank,df_meta3.SITE_ID,rotation=90)
ax.set_title(r"$R^2$ of GPP given observed g during growing season",fontsize=24)
fig.tight_layout()
fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.02),ncols=2)
#ax.vlines(df_meta.ddrain_mean,df_meta.tau_75,df_meta.tau_25,color="k")
#%%

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
    ax.text(0.1,0.8,  "$R^2$ = " + str(np.round(myr2,2)),transform=ax.transAxes)
#%%
fig,axes=plt.subplots(3,3,figsize=(12,10))

myplot(axes[0,0],df_meta.Aridity,df_meta.tau,
       "Annual aridity index",r"$\tau$ (days)")

myplot(axes[0,1],df_meta.Aridity_gs,df_meta.tau,
       "GS aridity index","")

myplot(axes[1,0],df_meta.map_data,df_meta.tau,
       "Annual P (mm/day)",r"$\tau$ (days)")

myplot(axes[1,1],df_meta.gsrain_mean,df_meta.tau,
       "GS P (mm/day)","")

axes[0,2].set_axis_off()

myplot(axes[2,2],df_meta.ddrain_2mean,df_meta.tau,
       "GS $D_{mean}$ (days)","")

myplot(axes[1,2],df_meta.gsrain_len,df_meta.tau,
       "GS length (days)","")


myplot(axes[2,1],df_meta.fullyear_dmax,df_meta.tau,
       "Annual $D_{max}$ (days)","")


myplot(axes[2,0],df_meta.fullyear_dmean,df_meta.tau,
       "Annual $D_{mean}$ (days)",r"$\tau$ (days)")

fig.tight_layout()

fig.legend(handles=points_handles,loc="upper center",bbox_to_anchor=(0.5,0.03),ncols=2 )
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
    ddlens = tabS.groupby("ddi").count().reset_index()
    #longDD = np.argmax(ddlens.et_per_F_dm)
    longDD = np.argmin(np.abs(ddlens.et_per_F_dm-20))
    
    jtab = tabS[tabS.ddi==longDD].reset_index()
    istart = np.where(np.isfinite(jtab.ET))[0][0]
    jtab = jtab.iloc[istart:].copy()
    tau = 20
    #sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    #epred20 = np.sqrt(2/tau*jtab.F2*(sm_init-jtab.etcum))
    term1 = -1/tau*jtab.F*jtab.G
    #c2 = (np.nanmean(jtab.ET) - np.nanmean(term1))/np.nanmean(jtab.F)
    c2 = jtab.ET.iloc[0]/jtab.F.iloc[0]
    # sm0 = 10
    # c1 = np.sqrt(sm0*4)
    # c2 = 0.5*c1*np.sqrt(2/tau)
    epred20 = np.clip(term1 + c2*jtab.F,0,np.inf)
    
    # sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    # smlist = []
    # etlist = []
    # s = 1*sm_init
    # for i in range(len(jtab)):
    #     smlist.append(s)
    #     eti = min(s,np.sqrt(2/tau*s*jtab.F2.iloc[i]))
    #     etlist.append(eti)
    #     s -= eti
    
    # epred20 = etlist
    
    #sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    #c1 = np.sqrt(sm_init*4)
    #sm_pred = 0.25*(-np.sqrt(2/tau)*jtab.G + c1)**2
    #epred20 = -np.diff(sm_pred)
    
    tau = 50
    term1 = -1/tau*jtab.F*jtab.G
    #c2 = (np.nanmean(ej) - np.nanmean(term1))/np.nanmean(f2)
    #c2 = (np.nanmean(jtab.ET) - np.nanmean(term1))/np.nanmean(jtab.F)

    c2 = jtab.ET.iloc[0]/jtab.F.iloc[0]
    #c2 = 0.5*c1*np.sqrt(2/tau)
    # sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    # c1 = np.sqrt(sm_init*4)
    # sm_pred = 0.25*(-np.sqrt(2/tau)*jtab.G + c1)**2
    # epred50 = -np.diff(sm_pred)
    
    epred50 =  np.clip(term1 + c2*jtab.F,0,np.inf)
    
    
    # sm_init = jtab.et2.iloc[0]/jtab.F2.iloc[0]/(2/tau)
    # smlist = []
    # etlist = []
    # s = 1*sm_init
    # for i in range(len(jtab)):
    #     smlist.append(s)
    #     eti = min(s,np.sqrt(2/tau*s*jtab.F2.iloc[i]))
    #     etlist.append(eti)
    #     s -= eti
    
    # epred50 = etlist
    
    
    #plt.subplot(2,1,si)
    xvar = np.arange(len(jtab.ET))+2
    plt.plot(xvar, np.array(jtab.ET),'ko-',linewidth=3,label="Eddy covariance")
    plt.plot(xvar,epred50,'o-',color="tab:blue",linewidth=3,alpha=0.6,label=r"Model, $\tau$ = 50 days")
    plt.plot(xvar,epred20,'o-',color="tab:orange",linewidth=3,alpha=0.6,label=r"Model, $\tau$ = 20 days")
    if si == 1:
        #plt.ylabel("ET (mm/day)",fontsize=22)
        #plt.ylim(-0.1,2)
        plt.legend(fontsize=16)
    # if si == 2:
        

    #     plt.xlabel("Day of drydown",fontsize=22)
    #     plt.ylabel("ET (mm/day)",fontsize=22)
    #plt.title(x)
    si += 1
#plt.tight_layout()
#plt.ylim(-0.1,3.9)
plt.xticks(np.arange(2,23,3))
plt.xlabel("Day of drydown",fontsize=22)
plt.ylabel("ET (mm/day)",fontsize=22)
plt.text(1.5,1.25,site_pair[0],fontsize=20)
plt.text(1.5,2.5,site_pair[1],fontsize=20)
plt.ylim(0,2.75)

# sinf = (ej / np.sqrt(fj) / np.sqrt(2/tau))**2
# s = sinf[0]
# sirec = np.zeros(len(fj))
# for i in range(len(fj)):
#     sirec[i] = s
#     eti = np.sqrt(2/tau*s*fj[i])
#     s -= eti
# etrec = np.sqrt(sirec * 2/tau * fj)
#%%

#%%
plt.figure(figsize=(10,8))
plt.axvline(0,color="grey",linestyle="--")
plt.axhline(0,color="grey",linestyle="--")

tab1 =  ddlist.loc[ddlist.SITE_ID=="US-Me5"].copy()
tab2 =  ddlist.loc[ddlist.SITE_ID=="US-SRM"].copy()
#tab2 =  ddlist.loc[ddlist.SITE_ID=="US-ARc"].copy()

plt.plot(tab1.row0,tab1.et_per_F_dm,'o',label=r"US-Me5, $\tau$ = 43 days",alpha=0.6)
plt.plot(tab2.row0,tab2.et_per_F_dm,'o',label=r"US-SRM, $\tau$ = 16 days",alpha=0.6)
rA = sm.OLS(tab1.et_per_F_dm,tab1.row0,missing='drop').fit()
rB = sm.OLS(tab2.et_per_F_dm,tab2.row0,missing='drop').fit()
xarr = np.array([-25,25])
plt.plot(xarr,xarr*rA.params[0],color="tab:blue")
xarr = np.array([-15,15])
plt.plot(xarr,xarr*rB.params[0],color="tab:orange")

plt.xlabel("Cumulative ET, daily value minus drydown mean (mm)")
plt.ylabel("$ET_{norm}$ = $ET^2/(VPD*g_A*LAI)$,\ndaily value minus drydown mean (mm/day)")
plt.legend(loc="lower left")
#%%

from statsmodels.stats.anova import anova_lm

biome_diff = anova_lm(smf.ols("tau_ddreg ~ C(combined_biome)",data=df_meta).fit())
