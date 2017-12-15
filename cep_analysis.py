import pandas as pd
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import struct
from matplotlib.patches import Circle

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

rEarth = 6371000 #radius of the earth in meters

fpath = input("enter path where data is located: ")
fpath = os.path.normpath(fpath)

filein = input("enter name of gps log file: ")

outplot = input("enter name for plot output: ")

center_lat = float(input("Enter latitude of CEP Center: "))
center_lon = float(input("Enter longitude of CEP Center: "))

cols = ['datetime','lat','lng','alt','valid']

df = pd.DataFrame()
df_raw = pd.DataFrame()

"""
for subdir,dirs,files in os.walk(fpath):
    for name in files:
        #print("merging files in dir " + os.path.join(fpath,name))
        print(os.path.join(fpath,name))
        try:
            wifi_df = pd.read_csv(os.path.join(fpath,name),header=None,names=wifi_cols)
            if df.empty:
                df = wifi_df
            else:
                df = pd.concat([df,wifi_df])
        except:
            print("Unable to merge file " + str(name))
"""
#uncomment below for windows
#df_raw = pd.read_csv(fpath+"\\"+filein,header=None,names=cols)
#uncomment below for linux
df_raw = pd.read_csv(fpath+"/"+filein,header=None,names=cols)

df_raw['datetime'] = pd.to_datetime(df_raw['datetime'],unit='s')

df = df_raw[df_raw['lat'].astype(str) != 'NOFIX']
df[['lat','lng','alt']] = df[['lat','lng','alt']].apply(pd.to_numeric, errors='coerce')


#center_lat = df['lat'].mean()
#center_lon = df['lng'].mean()


def describe_data(data):
    data = data[data != 0]
    data.dropna(inplace=True)
    print("count:   " + str(len(data)) )
    print("mean:    " + str(np.mean(data)) )
    print("std:     " + str(np.std(data)) )
    print("min:     " + str(np.min(data)) )
    print("max:     " + str(np.max(data)) )

#calcualte distance with law of Haversines
df['lat_rad'] = df['lat'].map(lambda x: math.radians(x))
df['lng_rad'] = df['lng'].map(lambda x: math.radians(x))

df['phi_deg'] = df['lat_rad'].sub(math.radians(center_lat))
df['lambda_deg'] = df['lng_rad'].sub(math.radians(center_lon))

df['sin'] = np.sin(df['phi_deg']/2)
df['cos'] = ((np.sin(df['phi_deg']/2)*(np.sin(df['phi_deg']/2)))) + ((np.cos(math.radians(center_lat)) * np.cos(df['lat_rad'])) * (np.sin(df['lambda_deg']/2) * np.sin(df['lambda_deg']/2)))

df['a'] = ((np.sin(df['phi_deg']/2) * (np.sin(df['phi_deg']/2)))) + ((np.cos(center_lat) * np.cos(df['lat_rad'])) * (np.sin(df['lambda_deg']/2) * np.sin(df['lambda_deg']/2)))

df['c'] = 2 * np.arctan2(np.sqrt(df['a']),np.sqrt(1-df['a']))

df['d'] = rEarth * df['c']

#calculate delta lat/lng
df['deltalat'] = df['lat_rad'].sub(math.radians(center_lat))
df['deltalng'] = df['lng_rad'].sub(math.radians(center_lon))
#assume points are close together anc cald distances/angles
#df['dy'] = np.sin(df['deltalng'])
#df['dx'] = math.cos(math.radians(center_lat))*(df['lng']-center_lon)
#df['dx'] = np.cos(math.radians(center_lat)) * np.sin(df['lat_rad']) - np.sin(math.radians(center_lat)) * np.cos(df['lat_rad']) * np.cos(df['deltalng'])

df['dy'] = df['deltalat']
df['dx'] = np.cos(math.radians(center_lat))*(df['deltalng'])

df['r'] = np.sqrt(np.power(df['dy'],2) + np.power(df['dx'],2))
df['rd'] = df['r'] * rEarth
df['angle'] = np.arctan2(df['dy'],df['dx'])
df['angle'] = df['angle'] + math.pi
df['angle_deg'] = df['angle'].map(lambda x : math.degrees(x))
df = df.sort_values(by='angle')

df['dy_m'] = np.sin(df['angle']) * df['rd']
df['dx_m'] = np.cos(df['angle']) * df['rd']
df.to_csv("test_out.csv")
#Calculate CEP Here
dx_extreme = df['dx_m'].abs().max()
dy_extreme = df['dy_m'].abs().max()
if dx_extreme > dy_extreme:
    ax_extreme = dx_extreme
else:
    ax_extreme = dy_extreme

sigma_dx = df['dx_m'].std()
sigma_dy = df['dy_m'].std()
radius_CEP = (0.56*sigma_dx) + (0.62*sigma_dy)
radius_2DRMS = 2*np.sqrt(sigma_dx**2 + sigma_dy**2)
print("CEP = " + str(radius_CEP))
print("2DRMS = " + str(radius_2DRMS))

#Draw Circles for Plotting
cir_CEP = plt.Circle((0,0),radius_CEP,color='forestgreen',alpha=0.4)
cir_2DRMS = plt.Circle((0,0),radius_2DRMS,color='tomato',alpha=0.4)
#patches.append(CEP_circle)
#patches.append(2DRMS_circle)

#custom legend definition
labels = [plt.Circle((0,0),0.005,color='forestgreen',alpha=0.4),plt.Circle((0,0),0.005,color='tomato',alpha=0.4),plt.Line2D((0,1),(0,0),color='dodgerblue',marker='o',linestyle='')]
descriptions = ['CEP = ' + str(radius_CEP),'2DRMS = ' + str(radius_2DRMS),'Fix Distance From True Position']
#p = PatchCollection(patches,alpha=0.4)

fig, ax1 = plt.subplots(figsize=(20,20))

sc=ax1.scatter(df['dx_m'],df['dy_m'],s=10,color='dodgerblue')

ax1.grid(True)
#ax1.legend(loc='upper left',prop={'size':'24'})
ax1.set_facecolor('whitesmoke')
ax1.add_artist(cir_2DRMS)
ax1.add_artist(cir_CEP)
ax1.legend(labels,descriptions,numpoints=1,markerscale=2,prop={'size': 24})
ax1.set_xlim([-ax_extreme,ax_extreme])
ax1.set_ylim([-ax_extreme,ax_extreme])
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Distance (m)')
ax1.xaxis.label.set_size(24)
ax1.yaxis.label.set_size(24)

axes = plt.gca()
#axes.set_xlim([0,360])
#axes.set_xticks([0,45,90,135,180,225,270,315,360])
#plt.colorbar(sc)
plt.title('GPS CEP',fontsize=24,weight='bold')
#uncomment below for windows
#plt.savefig(fpath+"\\"+outplot+".png",dpi=300)
#uncomment below for linux
plt.savefig(fpath+"/"+outplot+".png",dpi=300)

#df.to_csv(fpath+"\\"+outplot+"_wifi_df.csv")

