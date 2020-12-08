# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import array as arr 
import time as time 
from os import path
import matplotlib.pyplot as plt
import seaborn as sbn


prepared_data_file = "data/PeMS/prepared.pkl"

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

if not path.exists(prepared_data_file):
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
gamma

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


results_file = "output/results.pkl"

sp_list = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 9e-4, 13e-4]
l3_list = [10]
for i in range(30):
    start = time.time()
    print("i =", i)
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
            resdf.to_pickle(results_file)
            print(resdf.tail())
    end = time.time()
    print ("Time elapsed:", end - start)
    
# a1 = pd.read_pickle("output/res1.pkl")
# a2 = pd.read_pickle("output/res2.pkl")
# a3 = pd.read_pickle("output/res3.pkl")
# a4 = pd.read_pickle("output/res4.pkl")
# a5 = pd.read_pickle("output/res5.pkl")
# result = pd.concat([a1,a2,a3,a4,a5])
# result.to_pickle(results_file)

result = pd.read_pickle(results_file)
result
#result[result['name']=="omegaExt"].pivot(index='sparsity',columns='lambda3',values='unobservedMAPE')

pd.set_option('display.max_columns', None)
pd.pivot_table(result, values='unobservedMAPE', index=['name'], columns=['sparsity'], aggfunc=np.mean, fill_value=0)
result.replace({"omega": "FCD", "omega2x": "FCD x 2", "omegaExt" : "vision-xFCD"}, inplace=True)
prop = 'unobservedMAE'
grouped_df = result.groupby(['name','sparsity']).agg({prop:['mean', 'std', 'count'],
                                                      'coverage':'mean'}).reset_index()
grouped_df = grouped_df.assign(error = lambda x:1.96*x[prop]['std']/np.sqrt(x[prop]['count']))
grouped_df= grouped_df.assign(mv=lambda x:x[prop]['mean'],
                              lb = lambda x:x[prop]['mean']-x['error'],
                              ub = lambda x:x[prop]['mean']+x['error'],
                              cv = lambda x:x['coverage']['mean']*100)
svmap = grouped_df[grouped_df['name']=="FCD"][{"sparsity","cv"}]
svmap
grouped_df['sparsity']
svmap.set_index('sparsity')
grouped_df = grouped_df.join(svmap.set_index('sparsity'), on='sparsity', rsuffix='_omega')
grouped_df
groups = ["FCD","FCD x 2","vision-xFCD"]
my_dpi = 300
plt.rc('font', size=6)  
f, ax = plt.subplots(figsize=(2400/my_dpi,600/my_dpi), dpi=my_dpi)
sbn.lineplot(x="cv_omega", y="mv", hue="name",data=grouped_df, ax=ax,  hue_order=groups)

for group in groups:
    ax.fill_between(x=grouped_df.loc[grouped_df["name"] == group, "cv_omega"],
                    y1=grouped_df.loc[grouped_df["name"] == group, "lb"],
                    y2=grouped_df.loc[grouped_df["name"] == group, "ub"], alpha=0.5)
ax.set(xlabel='% of observed road segments', ylabel='Mean MAE, mph')
ax.legend(title='Data')
plt.show()
#f.savefig('test15.png', bbox_inches='tight')




