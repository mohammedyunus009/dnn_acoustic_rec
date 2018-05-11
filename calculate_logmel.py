import soundfile as sf
from scipy import signal
import librosa #package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems
import numpy as np
import config1 as cfg
def calculate_logmel(wav_fd, wr_fd):
    """Calculate log mel spectrogram 
    """
    names = [na for na in os.listdir(wav_fd) if na.endswith('.wav')]
    names = sorted(names)
    for na in names:
        
        path = os.path.join(wav_fd, na)
        wav, fs = sf.read(path)
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
        if globals().get('melW') is None:
            #print "done"
            global melW
            melW = librosa.filters.mel(sr=fs, 
                                       n_fft=cfg.n_fft, 
                                       n_mels=64, 
                                       fmin=0., 
                                       fmax=22100)
        x = np.dot(x, melW.T)
        x = np.log(x + 1e-8)
        
        #plt.matshow(x.T, origin='lower', aspect='auto')
        #plt.show()
        
        out_path = wr_fd + '/' + na[0:-4] + '.f'
        cPickle.dump(x, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL) #dumping the data ie processed into file 'out_path'
def to_mono(wav):
    if wav.ndim == 1: #ndim means dimention of array 1d or 2d
        return wav
    elif wav.ndim == 2:
        return np.mean(wav, axis=-1) 
  



if __name__ == "__main__":
    create_folder(cfg.dev_mel)
    create_folder(cfg.eva_mel)
    
    # calculate mel feature
    calculate_logmel(cfg.dev_wav, cfg.dev_mel) # takes wave file and destination to be written
    calculate_logmel(cfg.eva_wav, cfg.eva_mel)