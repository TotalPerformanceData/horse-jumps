# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:39:28 2019
***
Requires install of python environment of version supported by tensorflow/keras, model was trained in py3.6.
RetroModel predicts whether or not the point at time = t is a jump, from timestamps from time = t-8s to time = t+4s.
Suitable for restrospectively assessing whether a section was a jump and is usually more accurate than the live model
LiveModel predict whether or not the point at time = t is a jump, from timestamps from time = t-8s to time = t.
Inputs are at 2Hz, input to customeScaler is of shape (17,3), and uses first row as reference for bearing then removes.
Output of customScarler is (16,4), and can be fed into model.predict() as arr.reshape((1,16,4)). (observations, timesteps, features).

Other models could be trained on different intervals of time, different scaling schemes, different layer construction, 
additional defined parameters, different optimisation schemes, manufacturing extra training data applying random noise on existing data, 
different labelling scheme, removal of bypassed obstacles from training set, training models course by course...

Various info available on "freezing" the model for porting to C++, but this is not something i've done before and will only look into 
further if there's demand.
Loading the model takes a while, longest part of making predictions is the function call, timeit reports around 7ms+(0.07ms*observations).
eg for field of 10 horses (observations) the timer on my machine reports 7.7ms
***
@author: gswindells
email george.swindells@totalperformancedata.com
"""
from datetime import datetime, timedelta
import os, sys, json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd

def customScaler(X):
    # take nparray of [V, SF, Bearing], scale velocity using fixed max-min
    # output = np.array([velocity, SF, bearingChange, SL]) in [0,1]
    # reduces length in axis 0 by 1 for bearing benefit
    for i,x in enumerate(X): # handle SF nans
        if np.isnan(x[1]):
            if i > 0:
                X[i,1] = X[i-1,1]
            else:
                X[i,1] = np.mean(X[:,1])
    if np.any(np.isnan(X)):
        return False
    X = np.concatenate((X, np.zeros((X.shape[0],1))), axis=1) # initialise SL
    X = X[1:,:] # Remove initial row
    X[:,3] = ((X[:,0]/X[:,1]) - 3) / 5 # SL
    X[:,0] = (X[:,0] - 5) / 11 # V
    X[:,1] = (X[:,1] - 1.5) / 1.1 # SF
    X[:,2] = (X[:,2]-X[0,2]+3.141592653589793)%6.283185307179586 # initialise bearing change
    X[:,2] = (X[:,2] - min(X[:,2])) / 2 # Bearing
    X[np.abs(X) > 1] = 1. # replace values outside [0,1]
    X[np.abs(X) < 0] = 0.
    return X

def getBearing(coords1, coords2): 
    # return angle in radians to Meridean line for line between given coords in degrees
    lon1, lat1 = np.deg2rad(coords1)
    lon2, lat2 = np.deg2rad(coords2)
    return np.arctan2(np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1), np.sin(lon2-lon1)*np.cos(lat2))

def takeDatetime(ts):
    return datetime.strptime(ts[:-1].replace('T',' '), '%Y-%m-%d %H:%M:%S.%f')

def putDatetime(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f').replace(' ','T')[:-5]+'Z'

def fillLivePoints(raceDict, interval = timedelta(seconds=0.5)):
    # function to interpolate timestamps into model input format of 2Hz, input some dictionary of
    # timestamps, return new raceDict with timestamps interpolated at 'interval' steps.
    keys = sorted(list(raceDict.keys()), key=takeDatetime)
    start = takeDatetime(keys[0])
    end = takeDatetime(keys[-1])
    sc = raceDict[keys[0]]['I']
    daterange = pd.date_range(start=start, end=end, freq = interval)
    newKeys = [putDatetime(dt) for dt in daterange]
    for key in newKeys:
        if key not in raceDict:
            raceDict[key] = {'K':0,'I':sc,'T':key,'X':np.nan,'Y':np.nan,'V':np.nan,'P':np.nan,'SF':np.nan}
    df = pd.DataFrame.from_dict(raceDict, orient='index')
    df.sort_index(axis='index', inplace=True)
    df.interpolate(method='linear', limit_direction='forward', inplace=True)
    return df.to_dict('index')

if __name__ == '__main__':
    # For the file modelRetrospective.h5, timestamps of interval 0.5s are expected, of length 25 into customScalar (such that input to the model is of length 24)
    # 2010 shows a race in which the hurdles in the back straight were omitted due to low sun, in the subsequent race, 2040, all hurdles were jumped.
    modelRet = keras.models.load_model('modelRetroFinal.h5')
    modelLive = keras.models.load_model('modelLiveFenceFinal.h5')
    gpsPath = r'.\gpsFiles'
    if gpsPath not in sys.path:
        sys.path.append(gpsPath)
    gpsContents = os.listdir(gpsPath)

    for sc in gpsContents:
        if sc in gpsContents: 
            with open(os.path.join(gpsPath, sc), 'r') as file:
                gpsData = json.load(file)
            maxDist = max([row['P'] for row in gpsData])
            gpsDict = dict()
            for row in gpsData:
                if row['I'][-2:] not in gpsDict:
                    gpsDict[row['I'][-2:]] = dict()
                gpsDict[row['I'][-2:]][row['T']] = row
        else:
            continue
            
        jumpsPredDict = dict()
        for r in gpsDict:
            timestamps = list(gpsDict[r].keys())
            for i,ts in enumerate(gpsDict[r]):
                if any([gpsDict[rnum][ts]['P'] == 0. for rnum in gpsDict if ts in gpsDict[rnum]]):
                    break
                if i > 22 and len(timestamps) - i > 30 and gpsDict[r][ts]['P'] > 25:
                    indexes = [timestamps[j] for j in range(i-16, i+9, 1)] # +9 for retro, +1 for live
                    if gpsDict[r][indexes[0]]['V'] > 9 and gpsDict[r][ts]['V'] > 6.5:
                        arr = np.array([[gpsDict[r][j]['V'],gpsDict[r][j]['SF'], (getBearing([gpsDict[r][indexes[0]]['X'], gpsDict[r][indexes[0]]['Y']], [gpsDict[r][j]['X'], gpsDict[r][j]['Y']]))] for j in indexes], dtype=np.float64)
                        entry2 = customScaler(arr)
                        entry = customScaler(arr[:17])
                        #stationary = np.array([min(1, gpsDict[r][ts]['P']/100)])
                        if type(entry) is not bool: # for binary training
                            X = str(round(gpsDict[r][ts]['X'],5))
                            Y = str(round(gpsDict[r][ts]['Y'],5))
                            entry = np.array(entry).reshape(1,16,4)
                            entry2 = np.array(entry2).reshape(1,24,4)
                            predLive = modelLive.predict(entry)
                            predRet = modelRet.predict(entry2)
                            if X + ',' + Y not in jumpsPredDict:
                                jumpsPredDict[X + ',' + Y] = {'X':[], 'Y':[], 'Score':[], 'X2':[], 'Y2':[], 'Score2':[]}
                            jumpsPredDict[X + ',' + Y]['X'].append(gpsDict[r][ts]['X'])
                            jumpsPredDict[X + ',' + Y]['Y'].append(gpsDict[r][ts]['Y'])
                            jumpsPredDict[X + ',' + Y]['Score'].append(max(predLive[0][0], 0))
                            jumpsPredDict[X + ',' + Y]['X2'].append(gpsDict[r][ts]['X'])
                            jumpsPredDict[X + ',' + Y]['Y2'].append(gpsDict[r][ts]['Y'])
                            jumpsPredDict[X + ',' + Y]['Score2'].append(max(predRet[0][0], 0))
                   
        for key in jumpsPredDict:
            jumpsPredDict[key]['X'] = np.median(jumpsPredDict[key]['X'])
            jumpsPredDict[key]['Y'] = np.median(jumpsPredDict[key]['Y'])
            jumpsPredDict[key]['Score'] = np.median(jumpsPredDict[key]['Score'])
            jumpsPredDict[key]['X2'] = np.median(jumpsPredDict[key]['X2'])
            jumpsPredDict[key]['Y2'] = np.median(jumpsPredDict[key]['Y2'])
            jumpsPredDict[key]['Score2'] = np.median(jumpsPredDict[key]['Score2'])
        
        #more red => jump is predicted at that X,Y
        xaxis = []; yaxis = []; scores = []
        for key in jumpsPredDict:
            xaxis.append(jumpsPredDict[key]['X'])
            yaxis.append(jumpsPredDict[key]['Y'])
            scores.append(jumpsPredDict[key]['Score'])
        maxScore = np.max(scores)
        minScore = np.min(scores)
        scores = [[(s-minScore)/(maxScore-minScore),0,(maxScore-s)/(maxScore-minScore)] for s in scores]
        plt.figure(figsize=(12,8))
        plt.scatter(xaxis,yaxis,c=scores)
        plt.title(sc + ' Live Model, 8s time series no delay')
        plt.show()
        xaxis = []; yaxis = []; scores = []
        for key in jumpsPredDict:
            xaxis.append(jumpsPredDict[key]['X2'])
            yaxis.append(jumpsPredDict[key]['Y2'])
            scores.append(jumpsPredDict[key]['Score2'])
        maxScore = np.max(scores)
        scores = [[s/maxScore,0,(maxScore-s)/maxScore] for s in scores]
        plt.figure(figsize=(12,8))
        plt.scatter(xaxis,yaxis,c=scores)
        plt.title(sc + ' Retrospective Model, 12s time series, 4s delay')
        plt.show()
        