# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import array as arr 
import os
import pickle

data_dir = "data/PeMS"
prepared_data_file = "data/PeMS/preprocessed.pkl"

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

if not os.path.exists(prepared_data_file):
    df = pd.read_csv ('data/PeMS/d07_text_station_5min_2020_11_29.txt',header=None);
    meta = pd.read_csv ('data/PeMS/d07_text_meta_2020_11_16.txt',header=0,sep="\t");
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
    df = df.rename(columns={0: 'datetime', 1: 'station', 9:'volume', 10:'occupancy',11:'speed'})[['datetime','station','volume','occupancy','speed']]
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M:%S')
    df = df[df['station'].isin(stations)]
    df['speed'] = df['speed'].replace(np.nan,65)
    df['volume'] = df['volume'].replace(np.nan,0)
    df.to_pickle(prepared_data_file)
else:
    df = pd.read_pickle(prepared_data_file)
data_volume = df.pivot(index='datetime',columns='station',values='volume')
cr = data_volume.corr()
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

res = {'N':N,'T':T, 'data_speed':data_speed,'data_volume':data_volume,'Lw':Lw,'H':H, 'gamma':gamma}
pickle.dump(res, open(os.path.join(data_dir, "prepared.pkl"), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# from functions import randomOmega,omegaMAE,omegaMAPE, enhanceOmega, coverage,TGMCS
# omega = randomOmega(9e-4,data_volume)
# omega
# data_speed_mean = pd.DataFrame(np.nan, index=data_speed.index, columns=data_speed.columns)
# mean_vals=data_speed.mean() 
# for i in data_speed_mean.index:
#     data_speed_mean.at[i, :] = mean_vals
# data_speed_mean
 
# omega = randomOmega(1e-4,data_volume)
# omegaMAE(omega, data_speed, data_speed_mean)
# omegaMAPE(omega, data_speed, data_speed_mean)