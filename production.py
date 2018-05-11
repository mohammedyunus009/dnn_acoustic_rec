import soundfile as sf
import numpy as np
import librosa 
import config as cfg
from scipy import signal
import cPickle


def to_mono(wav):
    if wav.ndim == 1: 
        return wav
    elif wav.ndim == 2:
        return np.mean(wav, axis=-1) 


def calculate_logmel(rd_fd):

	wav, fs = sf.read(rd_fd)
	wav = to_mono(wav)
	#assert fs == cfg.fs
	ham_win = np.hamming(cfg.n_fft) 
	[f, t, x] = signal.spectral.spectrogram(x=wav, 
                                                window=ham_win, 
                                                nperseg=cfg.n_fft, 
                                                noverlap=0, 
                                                detrend=False, 
                                                return_onesided=True, 
                                                mode='magnitude') #Compute a spectrogram with consecutive Fourier transforms.
	x = x.T    
	print x.shape
	if globals().get('melW') is None:
		global melW
		melW = librosa.filters.mel(sr=fs, 
                                       n_fft=cfg.n_fft, 
                                       n_mels=64, 
                                       fmin=0., 
                                       fmax=22100)
           
	x = np.dot(x, melW.T)
	x = np.log(x + 1e-8)
	print x
    
	rd_fd +=".f"
	cPickle.dump(x, open(rd_fd , 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def make_pred(rd_path):
	calculate_logmel(rd_path)
	import kera_pred
	msg = kera_pred.others(rd_path+".f",cfg.ld_md)
	return msg

