import os
import sys

import librosa
import numpy as np
import yaml


SCALE_AUDIO = True

PREFIX = ".." if os.path.dirname(os.getcwd()) == "src" else ""

with open('src/exp/config.yml', 'r') as fd:
    exp_params = yaml.safe_load(fd)

RAW_DATA_PATH = os.path.join(PREFIX, exp_params['paths']['data']['raw'])


def getPitch(x,fs,winLen=0.02):
  p = winLen*fs
  frame_length = int(2**int(p-1).bit_length())
  hop_length = frame_length//2
  f0, voiced_flag, voiced_probs = librosa.pyin(y=x, fmin=80, fmax=450, sr=fs,
                                                 frame_length=frame_length,hop_length=hop_length)
  return f0,voiced_flag


def transformation(fileID):
    fs = None 
    x, fs = librosa.load(os.path.join(RAW_DATA_PATH, fileID),sr=fs)
    if SCALE_AUDIO: x = x/np.max(np.abs(x))
    f0, voiced_flag = getPitch(x,fs,winLen=0.02)
      
    power = np.sum(x**2)/len(x)
    voiced_fr = np.mean(voiced_flag)    

    return [power, voiced_fr]
