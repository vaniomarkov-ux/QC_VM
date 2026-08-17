# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:16:14 2024

@author: vanio
"""

import pandas as pd
#-----------------------------------------------------------------------------
# read zst datafile
#---------------------------------------------------------------------------
import zstandard as zstd
import io
from math import log
import numpy as np
import scipy.signal
from datetime import time
from arch import arch_model
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from pyinform.transferentropy import transfer_entropy
#from databento import decode_file
import pandas as pd
import databento_dbn as dbn
import databento as db
from databento import DBNStore
from io import BytesIO
from itertools import product
from collections import Counter, defaultdict
from typing import Sequence, Any


# publisher_id	uint16_t	The publisher ID assigned by Databento, which denotes the dataset and venue.
# instrument_id	uint32_t	The numeric instrument ID.
# ts_event	uint64_t	The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch.
# price	int64_t	The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001.
# size	uint32_t	The order quantity.
# action	char	The event action. Can be Add, Cancel, Modify, cleaR book, Trade, or Fill. See Action.
# side	char	The side that initiates the event. Can be Ask for the sell aggressor in a trade, Bid for the buy aggressor in a trade, or None where no side is specified by the original trade or the record was not a trade.
# flags	uint8_t	A bit field indicating event end, message characteristics, and data quality. See Flags.
# depth	uint8_t	The book level where the update event occurred.
# ts_recv	uint64_t	The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch.
# ts_in_delta	int32_t	The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv.
# sequence	uint32_t	The message sequence number assigned at the venue.
# bid_px_00	int64_t	The bid price at the top level.
# ask_px_00	int64_t	The ask price at the top level.
# bid_sz_00	uint32_t	The bid size at the top level.
# ask_sz_00	uint32_t	The ask size at the top level.
# bid_ct_00	uint32_t	The number of bid orders at the top level.
# ask_ct_00	uint32_t	The number of ask orders at the top level.
# 
#-----------------------------------------------------------------------------
def z_encode_series(series: pd.Series, alpha: float = 0.25, beta=1.0, std_type: str = 'ewm'):
    """
    Encode a real-valued time series into symbolic form using EWMA and std-based bins.

    Parameters:
    - series: input pandas Series
    - alpha: EWMA smoothing parameter (0 < alpha <= 1)
    - k: number of symbols (must be 4 for this specific binning logic)
    - std_type: 'ewm' or 'rolling' for std computation

    Returns:
    - symbols: pandas Series of integers in {0, 1, 2, 3}
    """

    # Step 1: EWMA mean
    ewma = series.ewm(alpha=alpha, adjust=False).mean()

    # Step 2: Std calculation
    if std_type == 'ewm':
        ewm_var = series.ewm(alpha=alpha, adjust=False).var(bias=False)
        std = np.sqrt(ewm_var)
    elif std_type == 'rolling':
        window = int(2 / alpha)
        std = series.rolling(window=window, min_periods=1).std()
    else:
        raise ValueError("std_type must be 'ewm' or 'rolling'")

    std = std.fillna(0.0)  # Replace NaNs with 0
    # Step 3: Construct bin edges for each time point
    lower = ewma - beta*std
    upper = ewma + beta*std

    # Vectorized binning
    values = series.values
    bins = np.full_like(values, fill_value=-1, dtype=int)

    bins[values < lower] = 0
    bins[(values >= lower) & (values < ewma)] = 1
    bins[(values >= ewma) & (values < upper)] = 2
    bins[values >= upper] = 3
    symbols = pd.Series(bins, index=series.index, name='symbol')
    symbol_counts = symbols.value_counts().sort_index()
    # Normalize to get proportions
    symbol_distribution = symbol_counts / symbol_counts.sum()
    #print("Symbol counts:\n", symbol_counts)
    print("\nSymbol distribution (proportions):\n", symbol_distribution)
    return symbols, ewma, lower, upper

#-----------------------------------------------------------------------------
def z_encoding(ts, alpha=0.01, alphastd =0.01):
    # Calcualting z_midprice OR (ts - df_ns['ret_ema'])/df_ns['ret_emstd'])
    ts_ewm   = ts.ewm(alpha=alpha, adjust=False).mean()
    ts_emstd = 1*ts.ewm(alpha=alphastd, adjust=False).std()
    z_ret     = np.where(ts <= (ts_ewm - ts_emstd), -2, 
                                  np.where(ts <= (ts_ewm - 0.01*ts_emstd), -1, 
                                  np.where( ((ts_ewm - 0.01*ts_emstd) <= ts) &
                                             ( ts <= (ts_ewm + 0.01*ts_emstd)),0,          
                                  np.where( ((ts_ewm + 0.01*ts_emstd)  < ts) &
                                             (ts <=  ts_ewm+ts_emstd),  1,
                                  np.where(ts >  ts_ewm+ts_emstd,  2, 0)))))

    return pd.Series(z_ret, index=ts.index, name='symbol')


#-----------------------------------------------------------------------------
def dbn_to_df(fName, fPath):
    file=fPath+"\\" + fName
    stored_data = db.DBNStore.from_file(file)
    df = stored_data.to_df()
    return df


def zstd_to_df(fName, fPath):
    file=fPath+"\\" + fName
    # Open the compressed .zst file in binary mode
    with open(file, 'rb') as compressed_file:
        # Create a ZstdDecompressor object
        decompressor = zstd.ZstdDecompressor()
        
        
        # Decompress the file in chunks
        with decompressor.stream_reader(compressed_file) as reader:
            # Read the decompressed data in chunks
            chunk_size = 2*16384  # Adjust the chunk size as needed
            buffer = io.StringIO()
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk.decode('utf-8'))        
        
        
        # Rewind the buffer and load it into a DataFrame
            buffer.seek(0)
            df = pd.read_csv(buffer)
       
    return df

def getFeaturesData(df, tStart,tEnd):
    
    zero_thr=0.0000001
    
    timeCol = 'timestamp'
    timeCol = 'ts_event'
    
    ask_prc_col = 'ask_price'
    ask_prc_col = 'ask_px_00'
    ask_sze_col = 'ask_sz_00'
    
    
    bid_prc_col = 'bid_price'
    bid_prc_col = 'bid_px_00'
    bid_sze_col = 'bid_sz_00'
    
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["ts_event"])
    df['time']      = pd.to_datetime(df['ts_event']).dt.time 
    #df = df.sort_values("timestamp").set_index("timestamp")
    df = df.set_index("timestamp").sort_index() 
    # Forward-fill quotes (state persistence)
    df[[bid_prc_col, ask_prc_col]] = df[[bid_prc_col, ask_prc_col]].ffill()

    # forward-fill book state first (important)
    df[[bid_sze_col, ask_sze_col, bid_prc_col, ask_prc_col]] = df[[bid_sze_col, ask_sze_col, bid_prc_col, ask_prc_col]].ffill()



    # Now we can drop rows that still have NaN (only the very beginning typically)
    df = df.dropna(subset=[bid_prc_col, ask_prc_col])

    
    # Work hours of NYSE
    
    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    
    #df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.between_time(timeSt, timeEn)
    

    
    # hour = ts.hour           # 8
    # minute = ts.minute       # 0
    # second = ts.second       # 0
    # microsecond = ts.microsecond  # 8154
    # nanosecond = ts.nanosecond    # 44044
    
    # first time stamp
    
    fts = df[timeCol].iloc[0]
    
    # first time stamp in seconds
    base = 60*60*fts.hour + 60 *fts.minute + fts.second
    
    
    # seconds from beginning (base) fro each row
    df['Secs'] = 60*60*df[ timeCol].dt.hour + 60*df[ timeCol].dt.minute + df[ timeCol].dt.second - base
    #df['Bucket'] = (df['Secs']/window).astype(int)
    df['mid_price'] = (df[ask_prc_col]+df[bid_prc_col])/2
    df['log_mid_price'] = np.log(df['mid_price'])  # this is the log - mid price at event lecvel

    
    df['bid_prc'] = df[bid_prc_col]
    df['ask_prc'] = df[ask_prc_col]
    
    df['bid_sze'] = df[bid_sze_col]
    df['ask_sze'] = df[ask_sze_col]    
    
    #df['mid_price_weighted'] =(df[ask_prc_col]*df[ask_sze_col] + df[bid_prc_col]*df[bid_sze_col])/(df[ask_sze_col] + df[bid_sze_col])
    
    
    df['time'] = pd.to_datetime(df['ts_event']).dt.time 
   
    sden =  df[bid_prc_col] + df[ask_prc_col]
    df['spread'] = np.where(sden == 0, 0.0, (df[bid_prc_col] - df[ask_prc_col]) / sden)
    df['log_spread'] = np.log(df['spread'])
        
    df['vol_weight'] = df[ask_sze_col]*df[ask_prc_col]+df[bid_sze_col]*df[bid_prc_col]
    df['vol']        = df[ask_sze_col]                +df[bid_sze_col]    
    
    
    #df['rltv_spread_price'] = (df[ask_prc_col]-df[bid_prc_col])/(df[ask_prc_col]+df[bid_prc_col]) 
    

    # imbalance as state
    den = df[ask_sze_col] + df[bid_sze_col]
    df['vol_imb'] = np.where(den == 0, 0.0, (df[bid_sze_col] - df[ask_sze_col]) / den)  # common sign: + when bid dominates





    # Innovation
    df['vol_imb_returns']   = (df['vol_imb']-df['vol_imb'].shift(1))

    df['vol_imb_innovation'] = np.where(
        (df['vol_imb'].shift(1) == 0) & (df['vol_imb'] == 0), 
        0,  # Case 1: If both current and previous vol_imb are 0, return 0
    
        np.where(
            (df['vol_imb'].shift(1) == 0) & (df['vol_imb'] != 0), 
            1,  # Case 2: If previous vol_imb was 0 but current is nonzero, return 1
    
            (df['vol_imb'] - df['vol_imb'].shift(1)) / df['vol_imb'].shift(1)  # Case 3: Normal innovation formula
        )
    )
    
    
    df['mid_price_shifted' ]   = df['mid_price'].shift(1)
    df['mid_price_returns']    = df['mid_price'].diff()
    df['mid_price_innovation'] = df['mid_price_returns']/df['mid_price_shifted'] 
    df['mid_price_movement']   = df['mid_price_returns'].apply(lambda x: 1 if x > zero_thr else -1 if x < -zero_thr else 0)
    df['log_mid_price_innovation'] = df['log_mid_price'].diff() 
    #df['rltv_spread_price_innovation'] = df['rltv_spread_price'].diff()
    df['mid_price_sqr_returns']    =  df['mid_price_returns']**2     
    df['mid_price_sqr_returns_innovation']    = df['mid_price_sqr_returns'].diff()
    
    return df
#-----------------------------------------------------------------------------
def resample(df, aggr_column, time_column,  freq, aggr):
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    fr = '10ms'
    df_regular = df[aggr_column].resample(fr).apply(aggr).ffill()
    if freq != fr:
        df_resampled = df_regular.resample(freq).apply(aggr).dropna().ffill().to_frame()
    else:    
        df_resampled = df_regular.to_frame()
        
    # Create a complete 1-second index covering the range of your data
    full_index= pd.date_range(start=df_resampled.index.min(), end=df_resampled.index.max(), freq=freq)
    # Reindex the DataFrame to the complete index and forward fill missing values
    return df_resampled.reindex(full_index).ffill()


#-----------------------------------------------------------------------------
# bucket content aggegation procedures
#-----------------------------------------------------------------------------
def aggregate_last(bucket):
    return(bucket[-1])    

def aggregate_avg(bucket):
    return sum(bucket)/len(bucket)    

def aggregate_wavg(bucket_price,bucket_volume):
    return sum(p * v for p, v in zip(bucket_price,bucket_volume))/sum(bucket_volume)    


def aggregate_fl(bucket):
    return (bucket[0]+bucket[-1])/2    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def getBuckets2(df, col,  minBucket = 10):

    current_bucket = -1
    x = [] # list of buckets
    y = [] # list of price lists per bucket
    mList = [] #list of values in the current bucket
    
    crVal = df[col][0]
    current_bucket =  df['Bucket'][0]
    mList = mList + [crVal]
    physical_bucket = current_bucket         #the physical buckets should be non-interrupted sequence 
    for i in range(1, len(df)):
        crVal = df[col][i]
           
        if df['Bucket'][i] == current_bucket : # in current bucket
            mList = mList + [crVal]
        else: #start a new bucket
            if len(mList) >= minBucket : #there are enough bucket elements
                x = x + [current_bucket]
                y = y + [mList]
                
            mList = [crVal]
            current_bucket = df['Bucket'][i]
            physical_bucket = physical_bucket+1
            if physical_bucket < current_bucket:
                #print('Gap')
                for j in range(physical_bucket,current_bucket):
                    x = x + [j]
                    y = y + [[]]
                physical_bucket = current_bucket
                    
    if len(mList) >= minBucket : #there are enough bucket elements
        x = x + [current_bucket]
        y = y + [mList]
    return x, y # since x is the bucket # we can link a date + time to it    
#-----------------------------------------------------------------------------

# get processes data for particular coliumn col from an initial frame:
# skip records where col doesn't change
# aggregarte inside buckets using specifid aggregation method
# future - populate/interpolate missing data/buckets
# returns lists of 
# ts - processed / aggregated / interpolated time series
# dts - innovations of ts
# dtsb - movement direction of the innovations 0/1
# secs - list of correponding seconds
#-----------------------------------------------------------------------------
def get_data_1 (df, timeSt, timeEn , col, aggregated = False, skip_no_change=True, aggregation_method='', z_thr=0):
    rdf = df.loc[df['time'].between(timeSt, timeEn) ]
    rdf = rdf.reset_index(drop=True)
    # irregularly spaced data from secdond start to second end 
    # ts should be the same as mp or vlt2
    if not aggregated: # event - level data
        ts   =  (df.loc[df['Secs'].between(timeSt, timeEn) ])[col].tolist()  
        # innovations of the time series same as mpr or dvlt
        dts  = [ts[i]-ts[i-1] for i in range(1, len(ts))] 
        dts = [0]+dts
        #dts  = [ts[i]-ts[i-1] for i in range(1, len(ts)) if ts[i] != ts[i-1] ] 
         
        # binary movements of the time series
        #dtsb  = [1 if ts[i]>ts[i-1] else 0 for i in range(1, len(ts))] 
        if skip_no_change:
            #dtsb  = [ sign01(dts[i]) for i in range(len(dts))]
            dtsb  = [1 if ts[i]>ts[i-1]+z_thr else 0 for i in range(1, len(dts))  if ts[i] != ts[i-1] ]
        else: # no change is 0 
            dtsb  = [1 if (ts[i]-ts[i-1]) > z_thr else -1 if (ts[i]-ts[i-1]) < -z_thr else 0 for i in range(1, len(dts))]
            dtsb = [0]+dtsb
            
        # filter the points with no md price mvoement
        
        # non-aggregated data to be returned ts, dts, dtsb
    
    # aggregated data
    skip_no_chage = False
    if aggregated:           # buckets per window seconds 
        x,y   = getBuckets2(rdf, col , minBucket = 0)
        xv,yv = getBuckets2(rdf,'vol', minBucket = 0)
        # x is list of buckets = x[i] is the value of bucket i; 
        # for eaxample if windows =1 1 bucket corresponds to 1 second and x[i] is the 
        # second during which the values in y [i] have been onserved
        # y is list of list of values; the list y[i] contains the values pbserved during bucket x[i]
        aggr = aggregate_last        #aggregteion method for the values in 1 bucket
        if aggregation_method=='avg':
            aggr = aggregate_avg
        elif aggregation_method=='wavg':
            aggr = aggregate_wavg

        # replace the empty buckets with the value of 
        # pevious bucket - i.e. the variable didn't change
        
        # aggregated time series: replies list of values in a bucket with its aggegation
        if aggregation_method =='wavg':
            ts = [aggr(bp,bv) if len(bp)>0 else None for bp,bv in zip(y,yv)] # the new time series is 1 value aggr(bucket)           
        else:    
            ts = [aggr(bucket) if len(bucket)>0 else None for bucket in y] # the new time series is 1 value aggr(bucket) 
        
        vl = [aggregate_avg(bucket) if len(bucket)>0 else None for bucket in yv]
        # here we need to populate the missing values, due to empty buckets
        prev = None
        ts = [
        (prev := value if value is not None else prev)
        for value in ts
        ]
        
        prev = None
        vl = [
        (prev := value if value is not None else prev)
        for value in vl
        ]
        
        # innovations of the time series same as mpr or dvlt
        dts  = [ts[i]-ts[i-1] for i in range(1, len(ts))] 
        dts  = [0]+dts
        
        # log returns
        ldts  = [log(ts[i]/ts[i-1]) for i in range(1, len(ts))] 
        ldts  = [0]+ldts
        
        # binary movements of the time series
        if skip_no_chage:
            #dtsb  = [ sign01(dts[i]) for i in range(len(dts))]
            dtsb  = [1 if dts[i]>0 else 0 for i in range(1, len(dts))  if dts[i] != 0.0 ]
        else: # no change is 0 
            dtsb  = [1 if dts[i]>0 else -1 if dts[i] < 0 else 0 for i in range(0, len(dts))] 
        
    return ts, dts, ldts, dtsb, x, vl 
#------------------------------------------------------------------------------
def discretize3(x):
    eps = 0.000001
    return -1 if x < -eps else (1 if x > eps else 0)
#------------------------------------------------------------------------------
def discretize2(x):
    eps = 0.0000001
    return -1 if x <= 0 else 1
#------------------------------------------------------------------------------
def discretize01(x):
    return 0 if x <= 0 else 1
#--------------------------------------------------------------------------------
# Class calculation
#------------------------------------------------------------------------------
def class_suff_average(prefixes, suffixes, pref_col, suff_col):
    return sum(suffixes)/len(suffixes)

# binary movement direction 1 step ahead
def class_suff_1_sign(prefixes, suffixes, pref_col, suff_col):
    return 0 if suffixes.iloc[0] <= 0.0 else 1

# EXCLUDED CLASS for 0
def class_price_up(prefixes, suffixes, pref_col, suff_col):
    up = sum(suff_col)/len(suff_col) - pref_col.iloc[-1]
    eps =0.0000001
    return 0 if up <= 0 else 1
    #return -1 if up < -eps else (1 if up > eps else 0)
    
def class_average_up(prefixes, suffixes):
    up = sum(prefixes)/len(prefixes) - sum(suffixes)/len(suffixes)
    #eps =0.0000001
    return 0 if up <= 0 else 1
    #return -1 if up < -eps else (1 if up > eps else 0)   
#------------------------------------------------------------------------------
def get_midprice_old(df, frq, aggr='last', bin_sym='-11'):
    # column is the name of feature to be used: 'mid_price',  'log_midprice'
    # class_column  is the feature used to calculate the class

    freq = str(frq) + 's'

    mp_1s = df["log_mid_price"].resample("1s").agg(aggr).ffill()
    
    if frq == 1:
        mp = mp_1s
    else:
        mp = mp_1s.resample(freq).agg(aggr).ffill()

    mpr = mp.diff()
    mpr.iloc[0]=0
    
    if bin_sym=='-11':
        mpb = mpr.apply(discretize2)
    if bin_sym=='01':
        mpb = mpr.apply(discretize01)
    if bin_sym=='-101':
        mpb = mpr.apply(discretize3)    
        
    # df_mpr = df_mpr.abs()

    return mp, mpr, mpb
#------------------------------------------------------------------------------
def get_midprice(df, frq, aggr='last'):
    # column is the name of feature to be used: 'mid_price',  'log_midprice'
    # class_column  is the feature used to calculate the class

    if frq == 'raw': # mp mpr  
        return df['log_mid_price'], df['log_mid_price_innovation']

    freq = frq #str(frq) + 's'

    mp_1s = df["log_mid_price"].resample("1s").agg(aggr).ffill()
    
    if frq == '1s':
        mp = mp_1s
    else:
        mp = mp_1s.resample(freq).agg(aggr).ffill()

    mpr = mp.diff()
    mpr.iloc[0]=0

    # different calculation approach 
    if frq == '1s':
        mp1 = mp
        mpr1 = mpr
    else:
        mp1 = df["log_mid_price"].resample(frq).agg(aggr).ffill()
        mpr1 = df['log_mid_price_innovation'].resample(frq).agg(aggr).ffill()
    return mp, mpr, mp1, mpr1
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_imbalance(df, frq,  aggr, bin_sym='-11'):
    # column is the name of feature to be used: 'mid_price',  'log_midprice'
    # class_column  is the feature used to calculate the class

    imb_1s = df["vol_imb"].resample("1s").agg(aggr).ffill()
    if frq == 1:
        imb = imb_1s
    else:
        imb = imb_1s.resample(str(frq)+"s").agg(aggr).ffill()
    
 
    imbr = imb.diff()
    
    imbr.iloc[0]=0
    if bin_sym=='-11':
        imbb = imbr.apply(discretize2)
 
    if bin_sym=='01':
        imbb = imbr.apply(discretize01)
        
    if bin_sym=='-101':
        imbb = imbr.apply(discretize3)        
        
    # df_mpr = df_mpr.abs()

    return imb, imbr, imbb
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_spread(df, frq,  aggr, bin_sym='-11'):
    # column is the name of feature to be used: 'mid_price',  'log_midprice'
    # class_column  is the feature used to calculate the class

    spr_1s = df["spread"].resample("1s").agg(aggr).ffill()
    if frq == 1:
        spr = spr_1s
    else:
        spr = spr_1s.resample(str(frq)+"s").agg(aggr).ffill()
       
    sprr = spr.diff()
    
    sprr.iloc[0]=0
    if bin_sym=='-11':
        sprb = sprr.apply(discretize2)
 
    if bin_sym=='01':
        sprb = sprr.apply(discretize01)
        
    if bin_sym=='-101':
        sprb = sprr.apply(discretize3)        

    return spr, sprr, sprb
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def get_feature(df, column, frq, dType, pSize, sSize, cFunc, class_column, ts_column = 'ts_event', aggr='mean', tStart = [9+5,30,0], tEnd=[15+5,  30, 0], returns = False):
    # column is the name of feature to be used: 'mid_price',  'log_midprice'
    # class_column  is the feature used to calculate the class
    # frq is integration frequency in seconds 1,10,...
    # dType is the disretization type: 'real', 'bin', 'z-encd'
    # pSize and sSize are the prefix/suffix sizes in discretization units
    # cFunc is the function estimating the class: cFunc(prefix, suffix)
    #-------------------------------------------------------------
    # other aggregation functions: last(), mean(), sum(), min(), max(), count(),
    # std(), var(), mean()
    #-------------------------------------------------------------
    # returns:
    # ts - time seriesas specified by date, column, frq, dType, returns and aggr
    # prefixes, suffixes: prefix and suffix for each point t in ts as specified by pSize, sSize. 
    # the data for point t is incuded in the prefix and not in the suffix
    # class assigned to each point t 
    # ts_column is the column with the time stamp
    
    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)


    # Replace first value if incorrect (NaN, -inf, inf) with 0
    if df.loc[0, column] in [np.nan, -np.inf, np.inf] or pd.isna(df.loc[0, column]):
        df.loc[0, column] = 0
        
    if df.loc[0, class_column] in [np.nan, -np.inf, np.inf] or pd.isna(df.loc[0, class_column]):
        df.loc[0, class_column] = 0
    
    # Replace all remaining invalid values (-inf, inf, NaN) with the previous valid value
    df[column]       = df[column].replace([np.inf, -np.inf], np.nan).ffill()
    df[class_column] = df[class_column].replace([np.inf, -np.inf], np.nan).ffill()    

    # ts can be bin or z version of ts_col

    freq = str(frq) + 's'
    df_ns  = resample(df, column ,ts_column, freq, aggr)        # The aggregated buckets start from second 0
    df_cns = resample(df, class_column ,ts_column, freq, aggr)  # The aggregated buckets start from second 0
    
    ts_col = df_cns[class_column]                               # copy of the original resampled/integrated time series      
    #plt_ts(log_returns_1s, shw = True, title="Log Mid-Price at 1 sec Returns for date "+str(date), label='Log Returns', color='red', xlabel='Time', ylabel='Log Returns')
    ts = df_ns[column]

    if returns:
        ts = ts.diff()
        ts.iloc[0]=0

    if dType == 'trn':
        ts = df_ns[column].apply(discretize3)
    
    if dType == 'bin':
        ts = ts.apply(discretize2)
        
    if dType == 'z_ret':
        # Calcualting z_midprice OR (ts - df_ns['ret_ema'])/df_ns['ret_emstd'])
        alpha = 0.01 # high alpha - better tracking; small alpha - smoothing
        alphastd = alpha
        df_ns['ret_ema']   = ts.ewm(alpha=alpha, adjust=False).mean()
        df_ns['ret_emstd'] = 1*ts.ewm(alpha=alphastd, adjust=False).std()
        df_ns['z_ret']     = np.where(ts <= (df_ns['ret_ema'] - df_ns['ret_emstd']), -2, 
                                      np.where(ts <= (df_ns['ret_ema'] - 0.01*df_ns['ret_emstd']), -1, 
                                      np.where( ((df_ns['ret_ema'] - 0.01*df_ns['ret_emstd']) <= ts) &
                                                 ( ts <= (df_ns['ret_ema'] + 0.01*df_ns['ret_emstd'])),0,          
                                      np.where( ((df_ns['ret_ema'] + 0.01*df_ns['ret_emstd'])  < ts) &
                                                 (ts <=  df_ns['ret_ema']+df_ns['ret_emstd']),  1,
                                      np.where(ts >  df_ns['ret_ema']+df_ns['ret_emstd'],  2, 0)))))

        ts = df_ns['z_ret']
        
    # Extract prefixes and suffixes from ts
    m = pSize      # first classified position (not time) in ts is m-1  
    n = sSize      # m-1 corresponds to time (second)  m if ts starts at sec=1 
    prefixes = [ts[t-m+1:t+1] for t in range(m-1, len(ts)-n)]
    suffixes = [ts[t+1:t+1+n] for t in range(m-1, len(ts)-n)]
    
    pref_col = [ts_col[t-m+1:t+1] for t in range(m-1, len(ts)-n)]
    suff_col = [ts_col[t+1:t+1+n] for t in range(m-1, len(ts)-n)]
    
    classes = [cFunc(p,s,pc,sc) for  (p,s,pc,sc) in zip(prefixes,suffixes,pref_col,suff_col)]
    return ts, prefixes, suffixes, classes 
#------------------------------------------------------------------------------
def estimate_subsequence_distributions(seq, max_subsequence_length):
    # Unique symbols in the sequence (for generating all possible subsequences)
    unique_symbols = sorted(set(seq))
    
    # Initialize storage for all subsequences and their probabilities
    all_subsequences = []
    distributions = []
    
    for length in range(1, max_subsequence_length + 1):
        # Generate all possible subsequences of current length
        possible_subseqs = list(product(unique_symbols, repeat=length))
        all_subsequences.append(possible_subseqs)
        
        # Count occurrences of each subsequence in the sequence
        subseq_counts = defaultdict(int)
        total_possible = max(0, len(seq) - length + 1)  # Avoid division by zero
        
        for i in range(len(seq) - length + 1):
            subseq = tuple(seq[i:i + length])
            subseq_counts[subseq] += 1
        
        # Compute probabilities (including unobserved subsequences)
        subseq_probs = []
        for subseq in possible_subseqs:
            count = subseq_counts.get(subseq, 0)
            prob = count / total_possible if total_possible > 0 else 0.0
            subseq_probs.append([subseq, prob])
        
        distributions.append(subseq_probs)
    
    return all_subsequences, distributions
#------------------------------------------------------------------------------
# efficient truncating version of estimate_subsequence_counts

def estimate_observed_subsequence_counts(
    seq: Sequence[Any],
    max_subsequence_length: int,
    sample_size: float = 1.0,
    sample_after_length: int = 2,
    random_state: int  = None,
    sort: str = "lexicographic",
    include_prob: bool = True,
):
    """
    Count only observed subsequences (non-zero frequency), and optionally
    subsample the observed support for longer lengths.

    Parameters
    ----------
    seq
        Input sequence (list, tuple, np.ndarray, pd.Series, ...).
    max_subsequence_length
        Maximum subsequence/window length.
    sample_size
        Float in [0, 1]. For lengths > sample_after_length, keep approximately
        this proportion of the non-zero-frequency subsequences.
        - 1.0 -> keep all observed subsequences
        - 0.5 -> keep about half
        - 0.0 -> keep none for those longer lengths
    sample_after_length
        Keep all observed subsequences for lengths <= sample_after_length.
        For lengths > sample_after_length, apply sampling with sample_size.
    random_state
        Seed for reproducible sampling.
    sort
        One of:
        - "lexicographic"
        - "count_desc"
        - "none"
    include_prob
        If True, each returned row is [subseq, count, total_possible, prob].
        If False, each row is [subseq, count, total_possible].

    Returns
    -------
    observed_subsequences : list[list[tuple]]
        Observed subsequences kept for each length.
    counts : list[list[list]]
        Per length, list of entries:
          [subseq, count, total_possible, prob]  if include_prob=True
          [subseq, count, total_possible]        if include_prob=False
    """
    if max_subsequence_length < 1:
        raise ValueError("max_subsequence_length must be >= 1.")
    if not (0.0 <= sample_size <= 1.0):
        raise ValueError("sample_size must be in [0, 1].")

    rng = np.random.default_rng(random_state)
    seq = list(seq)

    observed_subsequences = []
    counts = []

    n = len(seq)

    for length in range(1, max_subsequence_length + 1):
        total_possible = n - length + 1
        if total_possible <= 0:
            observed_subsequences.append([])
            counts.append([])
            continue

        # ---- count only observed subsequences
        subseq_counts = defaultdict(int)
        for i in range(total_possible):
            subseq = tuple(seq[i:i + length])
            subseq_counts[subseq] += 1

        items = list(subseq_counts.items())   # [(subseq, count), ...], only nonzero

        # ---- sorting
        if sort == "lexicographic":
            items.sort(key=lambda kv: kv[0])
        elif sort == "count_desc":
            items.sort(key=lambda kv: (-kv[1], kv[0]))
        elif sort == "none":
            pass
        else:
            raise ValueError("sort must be 'lexicographic', 'count_desc', or 'none'.")

        # ---- sample observed support only for longer lengths
        if length > sample_after_length and sample_size < 1.0:
            m = len(items)

            if sample_size == 0.0:
                items = []
            else:
                k = int(np.round(sample_size * m))
                k = min(max(k, 1), m)   # keep at least 1 if sample_size > 0 and m > 0

                chosen = rng.choice(m, size=k, replace=False)
                chosen.sort()  # preserve current order
                items = [items[idx] for idx in chosen]

        kept_subseqs = [subseq for subseq, _ in items]

        if include_prob:
            rows = [
                [subseq, count, total_possible, count / total_possible]
                for subseq, count in items
            ]
        else:
            rows = [
                [subseq, count, total_possible]
                for subseq, count in items
            ]

        observed_subsequences.append(kept_subseqs)
        counts.append(rows)

    return observed_subsequences, counts
#------------------------------------------------------------------------------
def estimate_subsequence_counts(seq, max_subsequence_length):
    # Unique symbols in the sequence (for generating all possible subsequences)
    unique_symbols = sorted(set(seq))
    
    # Initialize storage for all subsequences and their probabilities
    all_subsequences = []
    counts = []
    
    #-->>subseq_end_points = defaultdict(list)
    
    for length in range(1, max_subsequence_length + 1):
        # Generate all possible subsequences of current length
        possible_subseqs = list(product(unique_symbols, repeat=length))
        all_subsequences.append(possible_subseqs)
        
        # Count occurrences of each subsequence in the sequence
        subseq_counts = defaultdict(int)
        total_possible = max(0, len(seq) - length + 1)  # Avoid division by zero
        
        for i in range(len(seq) - length + 1):
            subseq = tuple(seq[i:i + length])
            subseq_counts[subseq] += 1
            
            # the time stamp of the end token in the sequence
            #-->>ts = seq[i:i + length].index[-1]
            #-->>subseq_end_points[subseq].append([ts])  
        
        # Compute probabilities (including unobserved subsequences)
        subseq_probs = []
        for subseq in possible_subseqs:
            count = subseq_counts.get(subseq, 0)
            #prob = count / total_possible if total_possible > 0 else 0.0
            #subseq_probs.append([list[subseq], count, total_possible])
            subseq_probs.append([subseq, count, total_possible])
        counts.append(subseq_probs)
    
    return all_subsequences, counts #-->>, subseq_end_points
#------------------------------------------------------------------------------
def get_data(fPath, date, column, frq, dType, pSize, sSize, cFunc, class_column, ts_column = 'ts_event', aggr='mean', tStart = [9+5,30,0], tEnd=[15+5,  30, 0], returns = False):
    # date is the calendar date encooded as yyyymmdd
    # column is the name of feature to be used: 'mid_price',  'log_midprice'
    # class_column  is the feature used to calculate the class
    # frq is integration frequency in seconds 1,10,...
    # dType is the disretization type: 'real', 'bin', 'z-encd'
    # pSize and sSize are the prefix/suffix sizes in discretization units
    # cFunc is the function estimating the class: cFunc(prefix, suffix)
    #-------------------------------------------------------------
    # other aggregation functions: last(), mean(), sum(), min(), max(), count(),
    # std(), var(), mean()
    #-------------------------------------------------------------
    # returns:
    # ts - time seriesas specified by date, column, frq, dType, returns and aggr
    # prefixes, suffixes: prefix and suffix for each point t in ts as specified by pSize, sSize. 
    # the data for point t is incuded in the prefix and not in the suffix
    # class assigned to each point t 
    # ts_column is the column with the time stamp
    
    window = frq                                   #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)
    filter_by_time = True
    
    if filter_by_time:      
        timeSt=  time( tStart[0],    tStart[1],  tStart[2])
        timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
        df = df.loc[df['time'].between(timeSt, timeEn) ]
        df = df.reset_index(drop=True)


    # Replace first value if incorrect (NaN, -inf, inf) with 0
    if df.loc[0, column] in [np.nan, -np.inf, np.inf] or pd.isna(df.loc[0, column]):
        df.loc[0, column] = 0
        
    if df.loc[0, class_column] in [np.nan, -np.inf, np.inf] or pd.isna(df.loc[0, class_column]):
        df.loc[0, class_column] = 0
    
    # Replace all remaining invalid values (-inf, inf, NaN) with the previous valid value
    df[column]       = df[column].replace([np.inf, -np.inf], np.nan).ffill()
    df[class_column] = df[class_column].replace([np.inf, -np.inf], np.nan).ffill()    

    # ts can be bin or z version of ts_col


    freq = str(frq) + 's'
    df_ns  = resample(df, column ,ts_column, freq, aggr)        # The aggregated buckets start from second 0
    df_cns = resample(df, class_column ,ts_column, freq, aggr)  # The aggregated buckets start from second 0
    
    ts_col = df_cns[class_column]                               # copy of the original resampled/integrated time series      
    #plt_ts(log_returns_1s, shw = True, title="Log Mid-Price at 1 sec Returns for date "+str(date), label='Log Returns', color='red', xlabel='Time', ylabel='Log Returns')
    ts = df_ns[column]

    if returns:
        ts = ts.diff()
        ts.iloc[0]=0

    if dType == 'trn':
        ts = df_ns[column].apply(discretize3)
    
    if dType == 'bin':
        ts = ts.apply(discretize2)
        
    if dType == 'z_ret':
        # Calcualting z_midprice OR (ts - df_ns['ret_ema'])/df_ns['ret_emstd'])
        alpha = 0.01 # high alpha - better tracking; small alpha - smoothing
        alphastd = alpha
        df_ns['ret_ema']   = ts.ewm(alpha=alpha, adjust=False).mean()
        df_ns['ret_emstd'] = 1*ts.ewm(alpha=alphastd, adjust=False).std()
        df_ns['z_ret']     = np.where(ts <= (df_ns['ret_ema'] - df_ns['ret_emstd']), -2, 
                                      np.where(ts <= (df_ns['ret_ema'] - 0.01*df_ns['ret_emstd']), -1, 
                                      np.where( ((df_ns['ret_ema'] - 0.01*df_ns['ret_emstd']) <= ts) &
                                                 ( ts <= (df_ns['ret_ema'] + 0.01*df_ns['ret_emstd'])),0,          
                                      np.where( ((df_ns['ret_ema'] + 0.01*df_ns['ret_emstd'])  < ts) &
                                                 (ts <=  df_ns['ret_ema']+df_ns['ret_emstd']),  1,
                                      np.where(ts >  df_ns['ret_ema']+df_ns['ret_emstd'],  2, 0)))))

        ts = df_ns['z_ret']
        
    # Extract prefixes and suffixes from ts
    m = pSize      # first classified position (not time) in ts is m-1  
    n = sSize      # m-1 corresponds to time (second)  m if ts starts at sec=1 
    prefixes = [ts[t-m+1:t+1] for t in range(m-1, len(ts)-n)]
    suffixes = [ts[t+1:t+1+n] for t in range(m-1, len(ts)-n)]
    
    pref_col = [ts_col[t-m+1:t+1] for t in range(m-1, len(ts)-n)]
    suff_col = [ts_col[t+1:t+1+n] for t in range(m-1, len(ts)-n)]
    
    classes = [cFunc(p,s,pc,sc) for  (p,s,pc,sc) in zip(prefixes,suffixes,pref_col,suff_col)]
    return ts, prefixes, suffixes, classes 
#------------------------------------------------------------------------------

def getVolatility1(fPath,date, frq, ts_column = 'ts_event', tStart = [9+5,30,0], tEnd=[15+5,  30, 0]):
    window = frq                                 #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)

    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)

    # we need the  RETURNS of the price: df['mid_price'] = df['log_mid_price']
    # at thois point the price
    # Aggregate the price are at irregular intervals inside 1 s
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    freq = '10ms'
    
    # take the mean of the price in every 10L interval
    aggr='mean'
    regularized = df['mid_price'].resample(freq).apply(aggr).ffill()
    
    # calculate the returns at the 10ms interval
    returns  = regularized.diff().dropna()
    lagged  = regularized.shift(1).dropna()
    
    innovations  =  returns/lagged
    
    # sqr the returns at 10L
    innovations = innovations**2
    
    # aggregate and sum the returns at the requested interval
    freq = str(frq) + 's'
    aggr = 'sum'
    returns = innovations.resample(freq).apply(aggr).ffill()
    
    # calculatye the returns at the 10L interval
    returns = np.sqrt(returns) 

    return returns

def getVolatility2(fPath,date, frq, ts_column = 'ts_event', tStart = [9+5,30,0], tEnd=[15+5,  30, 0]):
    window = frq                                 #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)

    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)

    # we need the  RETURNS of the price: df['mid_price'] = df['log_mid_price']
    # at thois point the price
    # Aggregate the price are at irregular intervals inside 1 s
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    freq = '10ms'
    
    # take the mean of the price in every 10L interval
    aggr='mean'
    df_regular = df['mid_price'].resample(freq).apply(aggr).ffill()
    
    # calculatye the returns at the 10L interval
    returns = df_regular.diff().dropna()
    
    # sqr the returns at 10L
    returns = returns**2
    
    # aggregate and sum the returns at the requested interval
    freq = str(frq) + 's'
    aggr = 'sum'
    returns = returns.resample(freq).apply(aggr).ffill()
    
    # calculatye the returns at the 10L interval
    returns = np.sqrt(returns) 

    return returns 
    


#------------------------------------------------------------------------------
def getVolatility3(fPath,date, frq, ts_column = 'ts_event', tStart = [9+5,30,0], tEnd=[15+5,  30, 0]):
    window = frq                                 #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)

    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)

    # we need the  RETURNS of the price: df['mid_price'] = df['log_mid_price']
    # at thois point the price
    # Aggregate the price are at irregular intervals inside 1 s
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    freq = '10ms'
    
    # take the mean of the price in every 10L interval
    aggr='mean'
    df_regular = df['mid_price'].resample(freq).apply(aggr).ffill()
    
    # now resample averaing again at the  requested frequency frq
    # 
    freq = str(frq) + 's'
    df_resampled = df_regular.resample(freq).apply(aggr).ffill().to_frame()
    
    # here we have the 'mid_price' at frq
    # Calcualte the returns:
    
    returns = df_resampled.diff().dropna()

    # Fit GARCH(1,1) model to returns
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")
    
    return  garch_fit.conditional_volatility    
# Here is another approach where the returns are calculated at the mili second level 
# and then are summed up to the desired frequency    
def getVolatility4(fPath,date, frq, ts_column = 'ts_event', tStart = [9+5,30,0], tEnd=[15+5,  30, 0]):
    window = frq                                 #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)

    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)

    # we need the  RETURNS of the price: df['mid_price'] = df['log_mid_price']
    # at thois point the price
    # Aggregate the price are at irregular intervals inside 1 s
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    freq = '10ms'
    
    # take the mean of the price in every 10L interval
    aggr='mean'
    df_regular = df['mid_price'].resample(freq).apply(aggr).ffill()
    
    # calculatye the returns at the 10L interval
    returns = df_regular.diff().dropna()
    
    
    # now resample at the requested frequency frqsumming up the returns in the interval
    # 
    freq = str(frq) + 's'
    aggr = 'sum' 
    returns = returns.resample(freq).apply(aggr).ffill().to_frame()
    
    # here we have the the (integrated) return at frq

    # Fit GARCH(1,1) model to returns
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_model.fit(disp="off")
    
    return  garch_fit.conditional_volatility  
    
#------------------------------------------------------------------------------
def tsrv(prices, J=5):
    prices = prices.dropna()
    log_prices = np.log(prices)
       
    # Fast RV
    returns_fast = np.diff(log_prices)
    n = len(returns_fast)
    RV_fast      = np.sum(returns_fast**2)
    
    #Slow RVs 
    slow_RVs = [] # slow realizations
    slow_LNs = [] # lengths of slow realizations
    
    for m in range(n-J+1):
        sample = log_prices[m::J]
        returns_slow = np.diff(sample)
        slow_RVs.append(np.sum(returns_slow**2))
        slow_LNs.append(len(returns_slow))
    
    RV_slow_avg = np.mean(slow_RVs)
    LN_slow_avg = np.mean(slow_LNs)
    
    tsrv_val = RV_slow_avg  - (LN_slow_avg / n) * RV_fast
    
    return tsrv_val
#------------------------------------------------------------------------------------------------------------------------
def tsmp_wrong(prices, J=5):
    prices = prices.dropna()
    # irreular log-prices
    log_prices = np.log(prices)
       
    # Fast MP

    MP_fast      = np.mean(log_prices)
    n = len(log_prices)
    #Slow RVs 
    slow_MPs = [] # slow realizations
    slow_LNs = [] # lengths of slow realizations
    
    for m in range(n-J+1):
        sample = log_prices[m::J]
        mp_slow = np.mean(sample)
        slow_MPs.append(mp_slow)
        slow_LNs.append(len(sample))
    
    MP_slow_avg = np.mean(slow_MPs)
    LN_slow_avg = np.mean(slow_LNs)
    
    tsmp_val = MP_slow_avg  - (LN_slow_avg / n) * MP_fast
    
    return tsmp_val

def tsrv_corrected(log_prices, J):
    n = len(log_prices) - 1
    if n < J: return np.nan
    
    # Fast RV (all data)
    RV_fast = np.sum(np.diff(log_prices)**2)
    
    # Slow RV (subsampled)
    RV_slow_sum = 0
    for i in range(J):
        sub_sample = log_prices[i::J]
        RV_slow_sum += np.sum(np.diff(sub_sample)**2)
    
    RV_slow_avg = RV_slow_sum / J
    
    # Bias adjustment factor
    adj = (n - J + 1) / (J * n)
    return RV_slow_avg - adj * RV_fast

def tsrv(prices: pd.Series, K: int = None, c: float = 1.0) -> float:
    """
    Two-Scale Realized Variance (Zhang, Mykland, Aït-Sahalia).
    prices: price level series within ONE estimation interval (already regularized)
    K: number of subgrids (a.k.a. J). If None, choose ~ c * n^(2/3).
    Returns: estimated integrated variance over the interval (not annualized).
    """
    p = prices.dropna().astype(float).values
    if p.size < 3:
        return np.nan

    x = np.log(p)
    r = np.diff(x)
    n = r.size  # number of increments
    
    # Fast RV (all data)
    RV_fast = np.sum(r * r)

    if K is None:
        # canonical scaling; tune c mildly (0.5..2) if you want
        K = int(np.floor(c * (n ** (2/3))))
    K = max(2, min(K, n))  # at most n

    slow_RVs = []
    slow_LNs = []

    # IMPORTANT: only K subgrids (offsets 0..K-1)
    for m in range(K):
        xm = x[m::K]
        if xm.size < 2:
            continue
        rm = np.diff(xm)
        slow_RVs.append(np.sum(rm * rm))
        slow_LNs.append(rm.size)

    if not slow_RVs:
        return np.nan

    RV_slow_avg = float(np.mean(slow_RVs))
    nbar = float(np.mean(slow_LNs))

    # TSRV integrated variance
    return RV_slow_avg - (nbar / n) * RV_fast


#---------------------------------------------------------------------------------------------------------------------------
def get_volatility(df, frequency, aggr='last'):                               #1          label every 1 second intervals

#   start with aggregation to 1s

    log_mid_1s = df["log_mid_price"].resample("1s").agg(aggr).ffill()
    r1 = log_mid_1s.diff()
    rv_1s = r1**2             # realized volatility at 1 sec
    
    if frequency == 1:
        return np.sqrt(rv_1s)

    # for non-overlapping bins
    rv = rv_1s.resample(str(frequency)+"s").sum()
    return np.sqrt(rv)

    # the irregular log mid-price is  df['log_mid_price']
    # regularize it at 30 or 50 msec interval 
    # we need the  RETURNS of the price: df['mid_price'] = df['log_mid_price']
    # at this point the price
    # Aggregate the price are at irregular intervals inside 1 s
   
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    reg_freq = '10ms'
    rf = 10
    reg_freq = '100ms'
    rf = 100
    num_samples = frequency*(1000//rf) # if rf are ms
    
    aggr = 'mean' # take the mean to represent ech freq interval
    # this is the stream of midprices atfrequency of 10ms
    mid_price_regularized = df['mid_price'].resample(reg_freq).apply(aggr).ffill()
        
    # Calculate the volatility as integration of the mid-price at given frequency  
    freq = str(frequency) + 's'
    
    # max_J ∈ {20, 30, 40, 50, 60}.
    max_J = 50
    J = max(5, min(max_J, num_samples // 10))
    
    volatility =  mid_price_regularized.resample(freq).apply(lambda price_interval: tsrv(price_interval, J))
                  #mid_price_regularized.resample(freq).apply(lambda price_interval: tsrv(price_interval, J=10*frq) )
    return(volatility )

#---------------------------------------------------------------------------------------------------------------------------
def getVolatilityTSRV(fPath,date, frq, ts_column = 'ts_event', tStart = [9+5,30,0], tEnd=[15+5,  30, 0]):
    window = frq                                 #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)

    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)

    # the irregular log mid-price is  df['log_mid_price']
    # regularize it at 0 msec interval
    # we need the  RETURNS of the price: df['mid_price'] = df['log_mid_price']
    # at thois point the price
    # Aggregate the price are at irregular intervals inside 1 s
   
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    freq = '10ms'
    aggr = 'mean' # take the mean to represent ech freq interval
    # this is the stream of midprices atfrequency of 10ms
    mid_price_regularized = df['mid_price'].resample(freq).apply(aggr).ffill()
        
    # Calculate the volatility as integration of the mid-price at given frequency frq
    freq = str(frq) + 's'
    volatility  = mid_price_regularized.resample(freq).apply(lambda price_interval:tsrv(price_interval, J=10*frq) )
    
    return(volatility )

def getMidPriceTSMP(fPath,date, frq, ts_column = 'ts_event', tStart = [9+5,30,0], tEnd=[15+5,  30, 0]):
    window = frq                                 #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)

    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)

    # the irregular log mid-price is  df['log_mid_price']
    # regularize it at 0 msec interval
    # we need the  RETURNS of the price: df['mid_price'] = df['log_mid_price']
    # at thois point the price
    # Aggregate the price are at irregular intervals inside 1 s
   
    df['timestamp'] =  pd.to_datetime(df['ts_event'])
    df.set_index('timestamp', inplace = True)
    
    # Reguarize at 10 ms    fr = '10L' - L is a milisecond 
    freq = '10ms'
    aggr = 'mean' # take the mean to represent ech freq interval
    # this is the stream of midprices atfrequency of 10ms
    mid_price_regularized = df['mid_price'].resample(freq).apply(aggr).ffill()
        
    # Calculate the volatility as integration of the mid-price at given frequency frq
    freq = str(frq) + 's'
    volatility = mid_price_regularized.resample(freq).apply(lambda price_interval:tsmp(price_interval, J=10*frq) )
    
    return(volatility)

def cross_correlation(x, y, max_lag):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    N = len(x)
    if N < 2:
        return np.arange(-max_lag, max_lag+1), np.full(2*max_lag+1, np.nan)

    x = x - x.mean()
    y = y - y.mean()
    sx = x.std(ddof=0)
    sy = y.std(ddof=0)
    if sx == 0 or sy == 0:
        return np.arange(-max_lag, max_lag+1), np.zeros(2*max_lag+1)

    c = scipy.signal.correlate(x, y, mode="full")  # length 2N-1
    lags = np.arange(-(N-1), N)

    # overlap count per lag
    overlap = (N - np.abs(lags)).astype(float)

    # convert to correlation coefficient
    c = c / (overlap * sx * sy)

    mid = N - 1
    return lags[mid-max_lag: mid+max_lag+1], c[mid-max_lag: mid+max_lag+1]
#-----------------------------------------------------------------------------
def xCorrelation(fPath,date, frq, col_p, col_v,aggr_p = 'mean', aggr_v = 'mean', ts_column = 'ts_event', tStart = [9+5,30,0], tEnd=[15+5,  30, 0]):
    # col_p - predictwed variable
    # col_s - variate
    window = frq                                 #1          label every 1 second intervals
    fName = 'xnas-itch-'+date+'.mbp-1.csv.zst'
    df = zstd_to_df(fName, fPath)
    df = getFeaturesData(df, window)

    timeSt=  time( tStart[0],    tStart[1],  tStart[2])
    timeEn=  time( tEnd[0]  ,    tEnd[1]  ,  tEnd[2] )
    df = df.loc[df['time'].between(timeSt, timeEn) ]
    df = df.reset_index(drop=True)

    # Aggregate the mid-price 
    freq = str(frq) + 's'

    X = resample(df, col_p ,ts_column, freq, aggr_p)
    Y = resample(df, col_v ,ts_column, freq, aggr_v)
    X = X.replace(-np.inf, -1)
    Y = Y.replace( np.inf,  1)


    # Set the maximum lag
    max_lag = 20
    
    
    
    # Calculate cross-correlation
    lags, corr = cross_correlation(X, Y, max_lag)
    
    # Calculate the confidence interval
    conf_level = 0.95
    conf_interval = stats.norm.ppf((1 + conf_level) / 2) / np.sqrt(len(X) - np.abs(lags))
    
    # Plot the cross-correlation with confidence interval
    plt.figure(figsize=(10, 6))
    plt.stem(lags, corr, linefmt='b-', markerfmt='bo', basefmt=' ', label='Cross-Correlation')
    plt.fill_between(lags, -conf_interval, conf_interval, color='red', alpha=0.3, label=f'{int(conf_level*100)}% Confidence Interval')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    plt.title(col_p+' vs '+col_v+ ' Frequency='+str(frq))
    plt.legend()
    plt.grid(True)
    plt.show()
    
def mi_time_series(series, max_lag=20, bins=20):
    mi_values = []
    lags = np.arange(1, max_lag + 1)

    for lag in lags:
        X = series[:-lag].values.reshape(-1, 1)
        Y = series[lag:].values
        mi = mutual_info_regression(X, Y, discrete_features=False, n_neighbors=5)
        mi_values.append(mi[0])

    return lags, mi_values

from scipy import ndimage

def transfer_entropy(X, Y, lag, base=2):
    """
    Compute TE(X -> Y) at a given lag using frequency counting.
    X, Y : array-like of symbolic (discrete) values, same length T
    lag  : positive integer
    """
    X, Y = np.array(X), np.array(Y)
    T = len(X)

    # Valid indices: t >= 1 (for y_{t-1}) and t >= lag (for x_{t-lag})
    t_start = max(1, lag)
    t_end   = T  # exclusive in Python slicing

    # Extract the three symbolic streams
    y_t      = Y[t_start:t_end]          # y_t
    y_prev   = Y[t_start-1:t_end-1]      # y_{t-1}
    x_lagged = X[t_start-lag:t_end-lag]  # x_{t-lag}

    N = len(y_t)

    # Count joint and marginal frequencies
    count_3  = defaultdict(int)   # (y_t, y_prev, x_lag)
    count_yy = defaultdict(int)   # (y_t, y_prev)
    count_yx = defaultdict(int)   # (y_prev, x_lag)
    count_y  = defaultdict(int)   # (y_prev,)

    for i in range(N):
        triple = (y_t[i], y_prev[i], x_lagged[i])
        count_3[triple]                      += 1
        count_yy[(y_t[i], y_prev[i])]        += 1
        count_yx[(y_prev[i], x_lagged[i])]   += 1
        count_y[y_prev[i]]                   += 1

    # Compute TE
    te = 0.0
    for (yt, yp, xl), c3 in count_3.items():
        p_joint  = c3 / N
        p_cond_xy = c3 / count_yx[(yp, xl)]          # p(y_t | y_prev, x_lag)
        p_cond_y  = count_yy[(yt, yp)] / count_y[yp] # p(y_t | y_prev)

        te += p_joint * np.log2(p_cond_xy / p_cond_y)

    return te


def transfer_entropy_profile(X, Y, lagmax, base=2):
    """Compute TE(X->Y) for lags 1..lagmax. Returns array of length lagmax."""
    return np.array([transfer_entropy(X, Y, lag, base) for lag in range(1, lagmax + 1)])


def transferEntropy(X,Y,delay,gaussian_sigma=None):
	'''
	TE implementation: asymmetric statistic measuring the reduction in uncertainty
	for a future value of X given the history of X and Y.
	Calculated through the Kullback-Leibler divergence with conditional probabilities

	Quantifies the amount of information from Y to X.
	args:
		- X (1D array):
			time series of scalars (1D array)
		- Y (1D array):
			time series of scalars (1D array)
	kwargs:
		- delay (int): 
			step in tuple (x_n, y_{n - delay}, x_(n - delay))
		- gaussian_sigma (int):
			sigma to be used
			default set at None: no gaussian filtering applied
	returns:
		- TE (float):
			transfer entropy between X and Y given the history of X
	'''
	if len(X)!=len(Y):
		raise ValueError('time series entries need to have same length')
	n = float(len(X[delay:])) 
	iqrx = stats.iqr(X)
	if iqrx == 0 or np.isnan(iqrx):
		iqrx = 1e-8  # or choose another fallback (like np.std(X))


	# number of bins for X and Y using Freeman-Diaconis rule
	# histograms built with numpy.histogramdd
	binX = int( (max(X)-min(X))
				/ (2* iqrx / (len(X)**(1.0/3))) )
    
    
	iqry = stats.iqr(Y)
	if iqry == 0 or np.isnan(iqry):
		iqry = 1e-8  # or choose another fallback (like np.std(X))

    
	binY = int( (max(Y)-min(Y))
				/ (2*iqry / (len(Y)**(1.0/3))) )

	# Definition of arrays of shape (D,N) to be transposed in histogramdd()
	x3 = np.array([X[delay:],Y[:-delay],X[:-delay]])
	x2 = np.array([X[delay:],Y[:-delay]])
	x2_delay = np.array([X[delay:],X[:-delay]])


	binX = min(binX, 64)
	binY = min(binY, 64)
    
	p3,bin_p3 = np.histogramdd(
		sample = x3.T,
		bins = [binX,binY,binX])

	p2,bin_p2 = np.histogramdd(
		sample = x2.T,
		bins=[binX,binY])

	p2delay,bin_p2delay = np.histogramdd(
		sample = x2_delay.T,
		bins=[binX,binX])

	p1,bin_p1 = np.histogramdd(
		sample = np.array(X[delay:]),
		bins=binX)

	# Hists normalized to obtain densities
	p1 = p1/n
	p2 = p2/n
	p2delay = p2delay/n
	p3 = p3/n

	# If True apply gaussian filters at given sigma to the distributions
	if gaussian_sigma is not None:
		s = gaussian_sigma
		p1 = ndimage.gaussian_filter(p1, sigma=s)
		p2 = ndimage.gaussian_filter(p2, sigma=s)
		p2delay = ndimage.gaussian_filter(p2delay, sigma=s)
		p3 = ndimage.gaussian_filter(p3, sigma=s)

	# Ranges of values in time series
	Xrange = bin_p3[0][:-1]
	Yrange = bin_p3[1][:-1]
	X2range = bin_p3[2][:-1]

	# Calculating elements in TE summation
	elements = []
	for i in range(len(Xrange)):
		px = p1[i]
		for j in range(len(Yrange)):
			pxy = p2[i][j]

			for k in range(len(X2range)):
				pxx2 = p2delay[i][k]
				pxyx2 = p3[i][j][k]

				arg1 = float(pxy*pxx2)
				arg2 = float(pxyx2*px)

				# Corrections avoding log(0)
				if arg1 == 0.0: arg1 = float(1e-8)
				if arg2 == 0.0: arg2 = float(1e-8)

				term = pxyx2*np.log2(arg2) - pxyx2*np.log2(arg1) 
				elements.append(term)

	# Transfer Entropy
	TE = np.sum(elements)
	return TE
# End TransferEntropy
#----------------------------------------------------------------------------
def te_counterfactual_discrete_old(
    X, Y, k=1, ly=1, lx=0,
    alpha=0.5, beta=0.5,
    missing=None
):
    """
    Counterfactual TE consistent with:
    compare factual p(y_t | C_t, x_{t-k}) vs counterfactual p_cf(y_t | C_t)
    where p_cf marginalizes out x_{t-k}.

    X, Y: 1D arrays of integer symbols (ideally 0..K-1). Pandas Series ok.
    k: lag for X_{t-k} -> Y_t
    ly: length of Y history in context C_t
    lx: optional length of X history in context (beyond the single x_{t-k})
    alpha: Dirichlet smoothing for p(y | C,x)
    beta:  Dirichlet smoothing for p(x | C)
    missing: if not None, skip samples where any involved value equals `missing`
    """
    X = np.asarray(X).astype(int).ravel()
    Y = np.asarray(Y).astype(int).ravel()
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same length")

    n = len(X)
    t0 = max(k + lx, ly)  # earliest t where context and x_{t-k} exist

    # Determine alphabet sizes (assumes non-negative symbols)
    if missing is not None:
        X_valid = X[X != missing]
        Y_valid = Y[Y != missing]
    else:
        X_valid, Y_valid = X, Y

    if len(X_valid) == 0 or len(Y_valid) == 0:
        return np.nan

    X_card = int(X_valid.max()) + 1
    Y_card = int(Y_valid.max()) + 1
    if X_card <= 0 or Y_card <= 0:
        return np.nan

    # Counts per context:
    # N_cx_y[c] is a (X_card, Y_card) matrix of counts N(c,x,y)
    # N_cx[c]   is a (X_card,) vector of counts N(c,x)
    N_cx_y = {}
    N_cx = {}
    N_total = 0

    def get_ctx(t):
        # C_t = (Y_{t-ly},...,Y_{t-1}) and optionally X_{t-k-lx},...,X_{t-k-1}
        y_part = tuple(Y[t-ly:t])
        if lx > 0:
            x_part = tuple(X[t-k-lx:t-k])
            return y_part + x_part
        return y_part

    for t in range(t0, n):
        xlag = X[t - k]
        ycur = Y[t]
        if missing is not None:
            if (xlag == missing) or (ycur == missing):
                continue
            if np.any(Y[t-ly:t] == missing):
                continue
            if lx > 0 and np.any(X[t-k-lx:t-k] == missing):
                continue

        c = get_ctx(t)

        if c not in N_cx_y:
            N_cx_y[c] = np.zeros((X_card, Y_card), dtype=np.int64)
            N_cx[c]   = np.zeros((X_card,), dtype=np.int64)

        if xlag < 0 or xlag >= X_card or ycur < 0 or ycur >= Y_card:
            continue

        N_cx_y[c][xlag, ycur] += 1
        N_cx[c][xlag] += 1
        N_total += 1

    if N_total == 0:
        return np.nan

    # Compute TE = sum_{c,x,y} p(c,x,y) log p(y|c,x) / p_cf(y|c)
    te = 0.0
    for c, M in N_cx_y.items():
        cx = N_cx[c].astype(float)  # shape (X_card,)
        c_sum = cx.sum()
        if c_sum <= 0:
            continue

        # p(x|c) with beta smoothing
        px_c = (cx + beta) / (c_sum + beta * X_card)  # shape (X_card,)

        # p(y|c,x) with alpha smoothing
        # row-normalize M over y
        row_sums = M.sum(axis=1).astype(float)  # (X_card,)
        # Avoid division by zero rows: smoothing handles it
        py_cx = (M + alpha) / (row_sums[:, None] + alpha * Y_card)  # (X_card, Y_card)

        # counterfactual p_cf(y|c) = sum_x p(y|c,x) p(x|c)
        pcf_y_c = (px_c[:, None] * py_cx).sum(axis=0)  # (Y_card,)

        # contribution: sum_{x,y} p(c,x,y) log2( p(y|c,x)/p_cf(y|c) )
        p_cxy = M / float(N_total)  # (X_card, Y_card)

        # Only where p_cxy > 0 matters
        nz = p_cxy > 0
        te += np.sum(p_cxy[nz] * (np.log2(py_cx[nz]) - np.log2(pcf_y_c[np.where(nz)[1]])))

    return float(te)

def te_counterfactual_discrete(X, Y, k=1, Ly=1, missing=-1, alpha=1.0):
    """
    Discrete counterfactual TE: feature X_{t-k} -> target Y_t
    conditioned on Y history of length Ly.
    Returns bits.
    """
    X = np.asarray(X).astype(np.int64).ravel()
    Y = np.asarray(Y).astype(np.int64).ravel()
    assert len(X) == len(Y)

    # infer alphabet sizes from observed symbols (excluding missing)
    Xm = X[X != missing]
    Ym = Y[Y != missing]
    if len(Xm) == 0 or len(Ym) == 0:
        return np.nan
    Kx = int(Xm.max()) + 1
    Ky = int(Ym.max()) + 1

    start = max(k, Ly)
    if len(X) - start < 10:
        return np.nan

    # encode Y history state h in base Ky
    powKy = (Ky ** np.arange(Ly, dtype=np.int64))  # Ky^0..Ky^(Ly-1)

    xs, ys, hs = [], [], []
    for t in range(start, len(X)):
        xk = X[t-k]
        yt = Y[t]
        yhist = Y[t-Ly:t][::-1]  # y_{t-1}..y_{t-Ly}
        if (xk == missing) or (yt == missing) or np.any(yhist == missing):
            continue
        h = int(np.dot(yhist, powKy))
        xs.append(int(xk)); ys.append(int(yt)); hs.append(h)

    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    hs = np.asarray(hs, dtype=np.int64)
    n = len(xs)
    if n < 50:
        return np.nan

    Kh = Ky ** Ly

    # counts with Dirichlet smoothing alpha
    # N(y,h,x)
    idx_yhx = ys*(Kh*Kx) + hs*Kx + xs
    c_yhx = np.bincount(idx_yhx, minlength=Ky*Kh*Kx).astype(float) + alpha
    c_yhx = c_yhx.reshape(Ky, Kh, Kx)

    # N(h,x)
    idx_hx = hs*Kx + xs
    c_hx = np.bincount(idx_hx, minlength=Kh*Kx).astype(float) + alpha
    c_hx = c_hx.reshape(Kh, Kx)

    # N(y,h)
    idx_yh = ys*Kh + hs
    c_yh = np.bincount(idx_yh, minlength=Ky*Kh).astype(float) + alpha
    c_yh = c_yh.reshape(Ky, Kh)

    # N(h)
    c_h = np.bincount(hs, minlength=Kh).astype(float) + alpha

    # probabilities
    p_yhx = c_yhx / c_yhx.sum()
    p_y_given_hx = c_yhx / c_hx[None, :, :]
    p_y_given_h  = c_yh  / c_h[None, :]

    ratio = p_y_given_hx / p_y_given_h[:, :, None]
    te = np.sum(p_yhx * np.log2(ratio))
    return float(te)


def te_curve_counterfactual_old(X, Y, k_max=40, **kwargs):
    """Compute TE_{X->Y}(k) for k=1..k_max."""
    return [te_counterfactual_discrete(X, Y, k=k, **kwargs) for k in range(1, k_max + 1)]

def te_curve_counterfactual(X, Y, k_max=40, Ly=1, missing=-1, alpha=1.0):
    return [te_counterfactual_discrete(X, Y, k=k, Ly=Ly, missing=missing, alpha=alpha)
            for k in range(1, k_max+1)]

#----------------------------------------------------------------------------
# Transfer Entropy - Single Variable
# steps_ahead - predictivity of values ahead
# lags - lags to calculate the predictablity

def single_var_transfer_entropy(X, steps_ahead=3, lags=20, sigma=None):
    te_list = []
    X = np.asarray(X)
    N = len(X)
    for pred in range(1, steps_ahead + 1):
        te_values = []
        for lag in range(1, lags + 1):
            # Build properly aligned time series
            x_past = X[:N - lag - pred]
            x_present = X[lag:N - pred]
            x_future = X[lag + pred:]
            
            if len(x_future) == 0:
                continue  # skip invalid lag/pred combo

            te_val = transferEntropy(
                X=x_future,
                Y=x_past,
                delay=1,
                gaussian_sigma=sigma
            )
            te_values.append(te_val)
        te_list.append(te_values)
    return te_list


colors = ['r', 'g', 'b', 'm', 'y']

def single_var_transfer_entropy_base(X, steps_ahead=3, lags = 20):
    te_list = []
    for pred in range(1, steps_ahead+1):  # Prediction steps ahead (1, 2, 3)
        te_values = []
        for lag in range(1, lags + 1):
            # Compute Transfer Entropy (Use your correct function)
            te_value = transferEntropy(X[:-pred], X[pred:], lag)
            te_values.append(te_value)
        te_list.append(te_values)
    return te_list

def plot_single_var_transfer_entropy(te_list, steps_ahead=3, lags = 20, title=''):
    colors = ['r', 'g', 'b', 'm', 'y']
    plt.figure(figsize=(8,5))
    for s in range(steps_ahead):
        plt.plot(range(1, lags+1), te_list[s],  linestyle='-', color=colors[s], label='Prediction '+str(s+1) +'-step ahead')
    
    # Formatting the plot
    plt.xlabel("Lag")
    plt.ylabel("Transfer Entropy")
    plt.title('Transfer Entropy '+ title)
    plt.legend()
    plt.grid()
    plt.show()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

def mutual_info(X, Y, bins=30):
    """
    Simple bivariate MI estimator via 2D histogram (equally spaced bins).
    Returns I(X; Y) in bits.
    """
    # Convert to float array
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    
    # 2D histogram
    hist_xy, x_edges, y_edges = np.histogram2d(X, Y, bins=bins)
    pxy = hist_xy / np.sum(hist_xy)
    
    px = np.sum(pxy, axis=1)  # marginal X
    py = np.sum(pxy, axis=0)  # marginal Y
    
    # Compute MI
    mi_value = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0.0:
                mi_value += pxy[i, j] * np.log2(pxy[i, j] / (px[i]*py[j]))
    return mi_value

#----------------------------------------------------------------------------

def time_lagged_mi(X, Y, max_lag=10, bins=30):
    """
    Computes MI(X_t, Y_{t+lag}) for lag in [-max_lag..max_lag].
    Returns:
      lags_array: array of lag values
      mi_array:   mutual info for each lag
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    lags_array = np.arange(-max_lag, max_lag+1, 1)
    mi_array = []
    
    for lag in lags_array:
        if lag >= 0:
            # Align X[0..end-lag] with Y[lag..end]
            X_sub = X[:-lag] if lag > 0 else X
            Y_sub = Y[lag:]
        else:
            # Align X[-lag..end] with Y[0..end+lag]
            X_sub = X[-lag:]
            Y_sub = Y[:lag]  # lag is negative => Y[:(-|lag|)]
        
        # Now compute bivariate MI
        mi_val = mutual_info(X_sub, Y_sub, bins=bins)
        mi_array.append(mi_val)
    
    return lags_array, np.array(mi_array)
#----------------------------------------------------------------------------
def space_time_mi(X, Y, max_lag_x=5, max_lag_y=5, bins=30):
    """
    Build a 2D matrix of MI for
    MI( X_{t - ell}, Y_{t + m} ), ell in [0..max_lag_x], m in [0..max_lag_y].
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    MI_mat = np.zeros((max_lag_x+1, max_lag_y+1))
    
    for ell in range(max_lag_x+1):
        for m in range(max_lag_y+1):
            # Align valid indices
            # We want pairs (t) such that t-ell >= 0 and t+m < len(X)
            # => t in [ell .. len(X)-1-m]
            t_indices = np.arange(ell, len(X)-m, 1)
            X_sub = X[t_indices - ell]
            Y_sub = Y[t_indices + m]
            
            # Compute MI
            MI_mat[ell, m] = mutual_info(X_sub, Y_sub, bins=bins)
    
    return MI_mat




#------------------------------------------------------------------------------

def X_pred_Y(X,Y, nameX, nameY, frequency, units, date, max_lag=20):
    
    txt = nameX +' vs '+ nameY + ' on '+ date[4:6]+'-'+date[6:8]+'-'+date[:4]
    # Calculate cross-correlation
    lags, corr = cross_correlation(X, Y, max_lag)
    
    # Calculate the confidence interval
    conf_level = 0.95
    conf_interval = stats.norm.ppf((1 + conf_level) / 2) / np.sqrt(len(X) - np.abs(lags))
    
    # Plot the cross-correlation with confidence interval
    plt.figure(figsize=(10, 6))
    plt.stem(lags, corr, linefmt='b-', markerfmt='bo', basefmt=' ', label='Cross-Correlation')
    plt.fill_between(lags, -conf_interval, conf_interval, color='red', alpha=0.3, label=f'{int(conf_level*100)}% Confidence Interval')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation:')
    plt.title('X Corr: '+txt+' '+str(frequency)+' '+units)
    plt.legend()
    plt.grid(True)
    plt.show()
    

#------------------------------------------------------------------------------
    # Compute time-lagged MI
    lags, mi_vals = time_lagged_mi(X, Y, max_lag, bins=20)

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(lags, mi_vals, marker='o')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("Lag (Y is shifted by lag)")
    plt.ylabel("MI(bits)")
    plt.title("MI:"+txt+' '+str(frequency)+' '+units)
    plt.grid(True)
    plt.show()
    
    mi_mat = space_time_mi(X, Y, max_lag_x=max_lag, max_lag_y=max_lag, bins=30)


    plt.figure(figsize=(8, 6))
    plt.imshow(mi_mat, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Mutual Information')
    
    # Axis labeling
    plt.xlabel('Future Lag of Y (m steps ahead)')
    plt.ylabel('Past Lag of X (ℓ steps back)')
    
    # Optional: Add custom tick labels if lags are known
    # lags = range(mi_mat.shape[0])
    # leads = range(mi_mat.shape[1])
    # plt.xticks(ticks=range(len(leads)), labels=leads)
    # plt.yticks(ticks=range(len(lags)), labels=lags)
    
    # Title
    plt.title('SpcTme MI:'+txt+' '+str(frequency)+' '+units)
    
    plt.tight_layout()
    plt.show()
    
#------------------------------------------------------------------------------
# transfer entropy

    te = [transferEntropy(X,Y,delay=k,gaussian_sigma=None) for k in range(1, max_lag + 1)] 
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_lag + 1), te, marker='o', linestyle='-', color='red')
    plt.title("TE OLD"+txt+' '+str(frequency)+' '+units)
    plt.xlabel("Lags (k)")
    plt.ylabel("Transfer Entropy")
    plt.grid(True)
    plt.show()

    
    te =  te_curve_counterfactual(X, Y, max_lag)
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_lag + 1), te, marker='o', linestyle='-', color='red')
    plt.title("TE NEW"+txt+' '+str(frequency)+' '+units)
    plt.xlabel("Lags (k)")
    plt.ylabel("Transfer Entropy")
    plt.grid(True)
    plt.show()
    
    te = transfer_entropy_profile(X, Y, max_lag, base=2) 

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_lag + 1), te, marker='o', linestyle='-', color='red')
    plt.title("TE OLD2 "+txt+' '+str(frequency)+' '+units)
    plt.xlabel("Lags (k)")
    plt.ylabel("Transfer Entropy")
    plt.grid(True)
    plt.show()


from scipy.stats import norm


def z_encode_ewm(
    series: pd.Series,
    n_symbols: int = 4,
    alpha: float = 0.05,
    min_periods: int = 50,
    clip_z: float = 4.0,
    sigma_scale=1,
    return_z: bool = False,
):
    """
    Exponentially-weighted z-encoding of a real-valued time series into
    discrete symbols {0, ..., n_symbols-1}.

    Steps:
    1. Compute EWM mean and std of the series.
    2. Compute z_t = (x_t - mu_t) / sigma_t.
    3. (Optionally) clip z to [-clip_z, clip_z].
    4. Discretize z using Gaussian-quantile bin edges so that each symbol
       has ~equal probability under N(0,1).

    Parameters
    ----------
    series : pd.Series
        Real-valued time series.
    n_symbols : int, default 4
        Number of discrete symbols.
    alpha : float, default 0.05
        EWM smoothing parameter. Larger alpha => faster adaptation.
    min_periods : int, default 50
        Minimum number of observations before EWM stats are considered valid.
        Earlier points will be NaN in the z-encoding.
    clip_z : float, default 4.0
        Clip z-scores to [-clip_z, clip_z] before binning.
    return_z : bool, default False
        If True, return a tuple (symbols, z_scores). Otherwise return symbols only.

    Returns
    -------
    symbols : pd.Series
        Discrete-encoded series with integer values in {0, ..., n_symbols-1}.
    z_scores : pd.Series (optional)
        The underlying z-scores, returned only if return_z=True.
    """

    s = series.astype(float).copy()

    # 1) EWM mean and variance
    ewm_mean = s.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
    ewm_var = s.ewm(alpha=alpha, adjust=False, min_periods=min_periods).var(bias=False)
    ewm_std = np.sqrt(ewm_var)

    # 2) z-scores
    z = (s - ewm_mean) / (ewm_std*sigma_scale)

    # Avoid crazy values from very small std
    z = z.replace([np.inf, -np.inf], np.nan)

    # 3) Clip z
    if clip_z is not None:
        z = z.clip(lower=-clip_z, upper=clip_z)

    # 4) Build Gaussian-quantile bin edges for N(0,1)
    #    e.g. for n_symbols=4, edges ~ [-inf, q0.25, q0.5, q0.75, +inf]
    probs = np.linspace(0, 1, n_symbols + 1)
    edges = norm.ppf(probs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Digitize: bin index in {0, ..., n_symbols-1}
    # np.digitize returns 1..n_symbols, so subtract 1
    symbols_array = np.digitize(z.to_numpy(), bins=edges[1:-1]) - 1

    symbols = pd.Series(symbols_array, index=series.index, name=f"sym_{n_symbols}")

    if return_z:
        z.name = "z_score"
        return symbols, z
    else:
        return symbols

#------------------------------------------------------------------------------

test = False
if test:    
    fName = 'xnas-itch-20250520.mbp-10.dbn.zst'
    fPath = 'C:\EXPIMP\Vanio\Projects\Market Data Preparation\Data'
    
    
    ts, prefixes, suffixes, classes  = get_data(fPath, date, column, frq, dType, pSize, sSize, cFunc, class_column, ts_column = 'ts_event', aggr='mean', tStart = [9+5,30,0], tEnd=[15+5,  30, 0], returns = False)
    
    
    #df = zstd_to_df(fName, fPath)
    df = dbn_to_df(fName, fPath)
    print(df.columns)
    print(df.head())
    print(len(df), 'rows')