"""
@Reference: https://github.com/smousavi05/STEAD; https://github.com/smousavi05/MagNet/blob/master/MagNet.py
"""

import numpy as np
import h5py
import pandas as pd
import torch
import torch.utils.data as Data


def string_convertor(dd):
    
    dd2 = dd.split()
    SNR = []
    for i, d in enumerate(dd2):
        if d != '[' and d != ']':
            
            dL = d.split('[')
            dR = d.split(']')
            
            if len(dL) == 2:
                dig = dL[1]
            elif len(dR) == 2:
                dig = dR[0]
            elif len(dR) == 1 and len(dR) == 1:
                dig = d
            try:
                dig = float(dig)
            except Exception:
                dig = None
            SNR.append(dig)
    return(SNR)


def get_data_stored(data_path):

    x_all=np.empty((0,3000,3))
    y_all=np.empty((0,4))
    print("starting reading data.....")

    for i in range(2,7):
        csv_file= data_path +r"chunk{}.csv".format(i)
        df = pd.read_csv(csv_file,low_memory=False) 
        common_codes = df.groupby('receiver_code').size()
        df = df[df['receiver_code'].isin(common_codes.index)]

        df = df[df.trace_category == 'earthquake_local']
        df = df[df.source_magnitude_type == 'ml']
        df = df[df.p_arrival_sample >= 200]
        df = df[df.p_arrival_sample+2900 <= 6000]
        df = df[df.p_arrival_sample <= 1500]
        df = df[df.s_arrival_sample >= 200]
        df = df[df.s_arrival_sample <= 2500]
        df.coda_end_sample = df.coda_end_sample.apply(lambda x: float(x.strip("[").strip("]")))
        df = df[df.coda_end_sample <= 3000]

        df.p_travel_sec.replace('None',np.nan, inplace = True)
        df = df[df.p_travel_sec.notnull()]
        df = df[df.p_travel_sec < 25] #Exclude extreme values

        df.source_distance_km.replace('None',np.nan, inplace = True)
        df = df[df.source_distance_km.notnull()]
        df = df[df.source_distance_km >= 0] 

        df.source_depth_km.replace('None',np.nan, inplace = True)
        df = df[df.source_depth_km.notnull()]
        df['source_depth_km'] = df['source_depth_km'].astype(float)
        df = df[df.source_depth_km>=0]

        df.source_magnitude.replace('None',np.nan, inplace = True)
        df = df[df.source_magnitude.notnull()]
        df = df[df.source_magnitude>=0]

        df.back_azimuth_deg.replace('None',np.nan, inplace = True)   
        df = df[df.back_azimuth_deg.notnull()]

        df.snr_db = df.snr_db.apply(lambda x: np.mean(string_convertor(x)))
        #df = df[df.snr_db >= 10] 

        
        f=h5py.File(data_path + r"chunk{}.hdf5".format(i),"r")
        count=0
        acc=f["data"]
        acckeyslist=list(acc.keys())
        dfindexlist=list(df.index)
        length=len(dfindexlist)
        x=np.zeros((length,3000,3),dtype=float)
        y=np.zeros((length,4),dtype=float)
        for index in dfindexlist:
            key= acckeyslist[index]
            accvalue=acc[key][()] 
            atr=acc[key].attrs
            mag=atr["source_magnitude"]
            epicenter=atr["source_distance_km"]
            depth = atr["source_depth_km"]
            p_travel_sec = atr['p_travel_sec']
            y[count][0]=mag
            y[count][1]=epicenter
            y[count][2]=p_travel_sec
            y[count][3]= depth
            par=int(atr["p_arrival_sample"])
            accvalue=accvalue[par-100:par+2900]
            # filteracc=bandpass(accvalue)
            x[count]=accvalue
            count+=1
        print("chunk{}".format(i))

        x_all=np.concatenate((x_all,x),axis=0)
        y_all=np.concatenate((y_all,y),axis=0)

    x_all = x_all.transpose(0,2,1)
    assert not np.any(np.isnan(x_all).any())
    assert not np.any(np.isnan(y_all).any())

    print(x_all.shape)
    print(y_all.shape)

    x_all = torch.from_numpy(x_all)
    y_all = torch.from_numpy(y_all)

    dataset=Data.TensorDataset(x_all,y_all)

    return dataset

