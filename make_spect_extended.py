import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
import librosa
from librosa.filters import mel
from numpy.random import RandomState
from pathlib import Path
import ipdb
from tqdm import tqdm

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
    
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
rootDir = './wavs'
# rootDir = './kids_speech/wav/'
# spectrogram directory
rootDirs = [
     '../data/LibriTTS/train-clean-100',
     '../data/kids_speech/wavs'
]
# rootDir = '/home/shacharm/Projects/ug/data/LibriTTS/train-clean-100'
# rootDir = '/home/shacharm/Projects/ug/data/kids_speech/wavs'
targetDir = './spmel'

for rootDir in rootDirs:
    assert Path(rootDir).exists(), "{} does not exist".format(rootDirs)
    
    dirName, subdirList, _ = next(os.walk(rootDir))
    print('Found directory: %s' % dirName)
    SAMPLE_RATE = 16000
    for subdir in tqdm(sorted(subdirList)):

        if False:
            files = (Path(rootDir) / subdir).glob('**/*.wav')

            if not os.path.exists(os.path.join(targetDir, subdir)):
                os.makedirs(os.path.join(targetDir, subdir))
            _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))

        try:
            prng = RandomState(int(subdir[1:])) 
        except:
            prng = RandomState()
        for fileName in tqdm(list((Path(rootDir) / subdir).glob('**/*.wav'))):
            
            targetSubDir = targetDir / fileName.relative_to(rootDir).parent
            targetSubDir.mkdir(parents=True, exist_ok=True)
            targetFile = (targetSubDir / fileName.stem).with_suffix('.npy')

            if targetFile.exists():
                continue

            # Read audio file
            #x, fs = sf.read(os.path.join(dirName,subdir,fileName))
            x, fs = sf.read(str(fileName))
            x = librosa.resample(x, fs, SAMPLE_RATE)
            fs = SAMPLE_RATE

            # Remove drifting noise
            y = signal.filtfilt(b, a, x)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            # Compute spect
            D = pySTFT(wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis)
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = np.clip((D_db + 100) / 100, 0, 1)    
            # save spect
            np.save(targetFile, S.astype(np.float32), allow_pickle=False)    
        
