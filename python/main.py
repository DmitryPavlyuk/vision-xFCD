# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import array as arr 
import time as time 

df = pd.read_csv ('data/PeMS/d07_text_station_5min_2020_11_29.txt',header=None);
print(df);

meta = pd.read_csv ('data/PeMS/d07_text_meta_2020_11_16.txt',header=0,sep="\t");
print(meta)

stations = arr.array('i', [
      717046,
      717045,
      717263,
      717264,
      716943,
      716942,
      716331,
      717445,
      717047,
      716028,
      716946,
      718085,
      718173,
      716939])

observable = {
      717046:717045,
      717045:717046,
      717263:717264,
      717264:717263,
      716943:716942,
      716942:716943,
      716331: 717445,
      717445: 716331,
      717047: 716028,
      716028:717047,
      716946: 718085,
      718085: 716946,
      718173: 716939,
      716939: 718173
    }
print(observable[716939])

df = df.rename(columns={0: 'datetime', 1: 'station', 9:'volume', 10:'occupancy',11:'speed'})[['datetime','station','volume','occupancy','speed']]
df.dtypes
df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M:%S')
print(df)
df = df[df['station'].isin(stations)]
df['speed'] = df['speed'].replace(np.nan,65)
df['volume'] = df['volume'].replace(np.nan,0)
data_volume = df.pivot(index='datetime',columns='station',values='volume')
cr = data_volume.corr()
-abs(cr)
len(cr.columns)
Lw = (-abs(cr)+np.identity(len(cr.columns)))/np.sum(cr, axis=1)

data_speed = df.pivot(index='datetime',columns='station',values='speed')

N = len(data_speed.columns)
T = len(data_speed)
H = np.zeros((T,T-1))
for i in range(T-1):
    H[i,i]=(-1)
    H[i+1,i]=1

gamma =  pd.DataFrame(np.identity(N), index=data_speed.columns, columns=data_speed.columns)

for key in observable:
    gamma.at[key, observable[key]] = 1
gamma

import matplotlib.pyplot as plt
plt.style.use('ggplot')
data_volume.plot.line()
data_speed.plot.line()


data_speed.std()

from functions import randomOmega,omegaMAE,omegaMAPE, enhanceOmega, coverage,TGMCS
omega = randomOmega(9e-4,data_volume)
omega
data_speed_mean = pd.DataFrame(np.nan, index=data_speed.index, columns=data_speed.columns)
mean_vals=data_speed.mean() 
for i in data_speed_mean.index:
    data_speed_mean.at[i, :] = mean_vals
data_speed_mean
 
omega = randomOmega(1e-4,data_volume)
omegaMAE(omega, data_speed, data_speed_mean)
omegaMAPE(omega, data_speed, data_speed_mean)

plt.matshow(omega)
coverage(omega)
    
# plt.matshow(omegaExt)
# coverage(omegaExt)
# omega.iloc[0]
# omegaExt.iloc[0]


sp = 5e-4

coverage(omega)
res = TGMCS(data_speed,Lw,H, omega, returnQhat = T, lambda3 = 1)
res['Qhat'].plot.line()
print(res['unobservedMAPE'],res['observedMAPE'],res['unobservedMAE'],res['observedMAE'])

results = {}




sp_list = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 9e-4, 13e-4]
l3_list = [10]
start = time.time()
for lambda3 in l3_list:
    for sp in sp_list:
        omega = randomOmega(sp,data_volume)
        omega2x = omega+randomOmega(sp,data_volume)
        omegaExt = enhanceOmega(omega, gamma)
        omegas = {'omega':omega, 'omega2x':omega2x, 'omegaExt':omegaExt}
        for o in omegas:
            print("lambda3 =", lambda3, "sp =", sp, "o = ", o)
            res = TGMCS(data_speed,Lw,H, omegas[o], lambda3 = lambda3)
            res['sparsity'] = sp
            res['lambda3'] = lambda3
            res['coverage'] = coverage(omegas[o])
            res['name'] = o
            results[len(results)+1] = res
        resdf = pd.DataFrame.from_dict(results, orient='index')
        resdf.to_pickle("output/results.pkl")
        print(resdf.tail())
end = time.time()
print ("Time elapsed:", end - start)

resdf = pd.DataFrame.from_dict(results, orient='index')
resdf[resdf['name']=="omegaExt"].pivot(index='sparsity',columns='lambda3',values='unobservedMAPE')
resdf.to_pickle("output/results.pkl")


