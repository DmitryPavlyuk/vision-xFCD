import pandas as pd
import numpy as np
import os
import pickle
from functions import composeMatrix

data_dir = "data/Simulated"

df = pd.read_excel( os.path.join(data_dir, "dataset_with_zones.xlsx"), engine='openpyxl')

df2=df.dropna(subset=['ZONE'])
df2["time"] = (pd.to_datetime(df['SIMTMOFDAY'], format='%H:%M:%S')- pd.Timedelta(minutes=5)).dt.round("min").dt.time
df2["ZONE"] = df2["ZONE"].astype('int64')

data_speed = composeMatrix(df2, "SPEED", np.mean)
data_speed = data_speed.rolling(window=6, min_periods=1).mean()
data_volume = composeMatrix(df2, "ID", len)

gamma = pd.read_excel( os.path.join(data_dir, "zones_observability.xlsx"), engine='openpyxl')
gamma.drop("Zones", axis=1,inplace=True)
connectivity = pd.read_excel( os.path.join(data_dir, "zones_connectivity.xlsx"), engine='openpyxl')
connectivity.drop("Zones", axis=1,inplace=True)

cr = data_volume.corr() #np.multiply(data_volume.corr(),connectivity)
Lw = (-abs(cr)+np.identity(len(cr.columns)))/np.sum(cr, axis=1)

N = len(data_speed.columns)
T = len(data_speed)
H = np.zeros((T,T-1))
for i in range(T-1):
    H[i,i]=(-1)
    H[i+1,i]=1

data_speed=data_speed.replace(0, np.nan).fillna(60, downcast='infer')
res = {'N':N,'T':T, 'data_speed':data_speed,'data_volume':data_volume,'Lw':Lw,'H':H, 'df':df2,'gamma':gamma}
pickle.dump(res, open(os.path.join(data_dir, "prepared.pkl"), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)