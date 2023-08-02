##Mel frequency energies computations parameters
window_length = 30 #in ms
window_step = 10   #in ms
n_mels = 40 #Number of mel bands for the mel frequency energies       

##Audio 
sr = 16000 #sampling rate 
#Number of spectrogram frames in a partial utterance
partials_n_frames = 160 # in # of frames, same as hop_length = sr * window_step / 1000
#t = partials_n_frames * step = 160 * 10 / 1000 = 1.6s

##Voice Activity Detector VAD
vad_window_length = 30 #in ms, should be eiter 10, 20 or 30 
#Number of frames to average together when performing the moving average smoothing, 
#the larger this value, the larger the VAD variations must be to not get smoothed out 
vad_moving_average_width = 8 
#Maximum number of consecutive silent frames a segment can have
vad_max_silence_length = 6

##Audio volume normalization 
audio_norm_target_dbFS = -30 #dB relative to full scale