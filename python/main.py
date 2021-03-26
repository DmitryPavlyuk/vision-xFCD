import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import time as time 
import seaborn as sbn
from functions import randomOmega, simulatedOmega, enhanceOmega, coverage,TGMCS

ds_name = "Simulated"
data_dir = os.path.join("data", ds_name)
data = pickle.load(open(os.path.join(data_dir, "prepared.pkl"), 'rb'))

isSimulated = True if ("df" in data) else False

data["data_speed"].plot(subplots=True,figsize=(8,16))
plt.savefig(os.path.join("output", ds_name+'-all-speeds.png'), dpi=300)
plt.show()

sp = 5e-3
omega = simulatedOmega(sp,data["df"], "SPEED") if isSimulated else randomOmega(sp,data["data_volume"])
coverage(omega)
plt.matshow(data["data_speed"])
plt.matshow(np.multiply(omega, data["data_speed"]))

res = TGMCS(data["data_speed"],data["Lw"],data["H"], omega,returnQhat = True, lambda3=10)
print(res['unobservedMAPE'],res['observedMAPE'],res['unobservedMAE'],res['observedMAE'])

mres = pd.concat([data["data_speed"],  res["Qhat"]], axis=1)
mres.columns=["real"+str(s) for s in data["data_speed"].columns]+["est"+str(s) for s in res["Qhat"].columns]
mres.iloc[:, [1, 1+len(res["Qhat"].columns)]].plot.line()

results_file = os.path.join("output", "results_"+ds_name+".pkl")
results = pd.read_pickle(results_file)
sp_list = [5e-4, 1e-3,2e-3,3e-3, 5e-3, 1e-2]
# for PeMS sp_list = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 9e-4, 13e-4]
l3_list = [10]
for i in range(20):
    start = time.time()
    print("i =", i)
    for lambda3 in l3_list:
        for sp in sp_list:
            omega = simulatedOmega(sp,data["df"], "SPEED") if isSimulated else randomOmega(sp,data["data_volume"])
            omega2x = simulatedOmega(sp*2,data["df"], "SPEED") if isSimulated else randomOmega(sp*2,data["data_volume"])
            omegaExt = enhanceOmega(omega.to_numpy(), data["gamma"].to_numpy())
            omegas = {'omega':omega, 'omega2x':omega2x, 'omegaExt':omegaExt}
            for o in omegas:
                print("lambda3 =", lambda3, "sp =", sp, "o = ", o)
                res = TGMCS(data["data_speed"],data["Lw"],data["H"], omegas[o], lambda3 = lambda3)
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
  
resdf.groupby(["name", "sparsity"]).size()

result = pd.read_pickle(results_file)
mae_units = 1
#mae_units = 1.60934

result["observedMAE"] = result["observedMAE"]*mae_units
result["unobservedMAE"] = result["unobservedMAE"]*mae_units
result.groupby(["name", "sparsity"]).size()
result

pd.set_option('display.max_columns', None)
pd.pivot_table(result, values='unobservedMAPE', index=['name'], columns=['sparsity'], aggfunc=np.mean, fill_value=0)
pd.pivot_table(result, values='unobservedMAE', index=['name'], columns=['sparsity'], aggfunc=np.mean, fill_value=0)

result.replace({"omega": "FCD", "omega2x": "FCD x 2", "omegaExt" : "vision-xFCD"}, inplace=True)
prop = 'unobservedMAE'
grouped_df = result.groupby(['name','sparsity']).agg({prop:['mean', 'std', 'count'],'coverage':'mean'}).reset_index()
grouped_df
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

grouped_df = grouped_df[grouped_df["cv_omega"]<25]
groups = ["FCD","FCD x 2","vision-xFCD"]
my_dpi = 300
plt.rc('font', size=6)  
f, ax = plt.subplots(figsize=(1200/my_dpi,600/my_dpi), dpi=my_dpi)
sbn.lineplot(x="cv_omega", y="mv", hue="name",data=grouped_df, ax=ax,  hue_order=groups)
ax.xaxis.set_major_locator(plt.MaxNLocator(7))
for group in groups:
    ax.fill_between(x=grouped_df.loc[grouped_df["name"] == group, "cv_omega"],
                    y1=grouped_df.loc[grouped_df["name"] == group, "lb"],
                    y2=grouped_df.loc[grouped_df["name"] == group, "ub"], alpha=0.5)
ax.set(xlabel='% of observed road segments', ylabel='Mean MAE, kmh')
ax.legend(title='Data')
plt.show()
f.savefig(os.path.join("data", ds_name+'-MAE.png'), bbox_inches='tight')
