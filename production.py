import soundfile as sf
import numpy as np
import librosa
import wavio 
import config1 as cfg
from scipy import signal
import cPickle


def to_mono(wav):
    if wav.ndim == 1: #ndim means dimention of array 1d or 2d
        return wav
    elif wav.ndim == 2:
        return np.mean(wav, axis=-1) 

def readwav(path):
    Struct = wavio.read(path) #/Wav(data.shape=(1323001, 2), data.dtype=int32, rate=44100, sampwidth=3)    Struct is just a varialble
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1) #/home/ruksana/test/DCASE2016_Task1/TUT-acoustic-scenes-2016-development/audio
    fs = Struct.rate #/44100 (all files same output)
    return wav, fs


# data, samplerate = sf.read('b010_0_30.wav')
def calculate_logmel(rd_fd):

	wav, fs = sf.read(rd_fd)
	wav = to_mono(wav)
	assert fs == cfg.fs
	ham_win = np.hamming(cfg.n_fft) #its giving me a bell curve
	[f, t, x] = signal.spectral.spectrogram(x=wav, 
                                                window=ham_win, 
                                                nperseg=cfg.n_fft, 
                                                noverlap=0, 
                                                detrend=False, 
                                                return_onesided=True, 
                                                mode='magnitude') #Compute a spectrogram with consecutive Fourier transforms.
	x = x.T     # (n_frames, n_freq)
	print x.shape
        # Mel transform matrix
	if globals().get('melW') is None:
			#print "done"
		global melW
		melW = librosa.filters.mel(sr=fs, 
                                       n_fft=cfg.n_fft, 
                                       n_mels=64, 
                                       fmin=0., 
                                       fmax=22100)
            #Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
		print "Efef"
		print melW.shape #(64, 513)
         #melW /= np.max(melW, axis=-1)[:,None]
         #Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
           
	x = np.dot(x, melW.T)#Dot product of two arrays. For 2-D arrays it is equivalent to matrix multiplication, and for 1-D arrays to inner product of vectors 
	x = np.log(x + 1e-8)#The natural logarithm log is the inverse of the exponential function, so that log(exp(x)) = x.
	print x
    #out_path = fe_fd + '/' + na[0:-4] + '.f'
	rd_fd +=".f"
	cPickle.dump(x, open(rd_fd , 'wb'), protocol=cPickle.HIGHEST_PROTOCOL) #dumping the data ie processed into file 'out_path'           


def make_pred(rd_path):
	calculate_logmel(rd_path)
	import kera_pred
	msg = kera_pred.others(rd_path+".f",cfg.wr_pred,cfg.ld_md)
	return msg

