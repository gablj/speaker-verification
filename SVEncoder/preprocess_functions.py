"""
This module defines the functions that are used by 'data_preprocess.py' 
to preprocess the audiofiles into .npy files that contain
the mel-filterbank energies of the waveforms.
"""
from pathlib import Path
from typing import Union, Optional 

import numpy as np
import librosa 
import webrtcvad 
from struct import pack #"pack()" takes non-byte values and converts them to bytes, "unpack()" converts from bytes to the specified equivalent
from scipy.ndimage import binary_dilation

from SVEncoder import params_data

def preprocess_wav(fpath_or_wav: Union[Path, str, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize_vol: Optional[bool] = True, 
                   trim_silence: Optional[bool] = True) -> np.ndarray:
    """
    Takes a string path to a waveform/soundfile or a waveform and calls the corresponding 
    functions to preprocess the waveform and returns it ready to be used to compute 
    the mel-spectrogram/mel-filterbank energies. 

    Parameters
    ----------
    fpath_or_wav : Path, str or np.ndarray
        A string path, a 'pathlib.Path' instance to the location of the soundfile
        or the waveform as a numpy array of floats.
    source_sr : int, optional, default = None
        if passing a waveform, can pass the "original" sampling rate of the waveform,
        after processing the sampling rate will match the one from the data hyperparameters. 
    normalize_vol : bool, optional, default = True
        If "True" normalizes the signals amplitud relative to
        how far bellow is from its peak amplitude. 
    trim_silece : bool, optional, default = True
        If "True", trim long silences from the waveform. 

    Returns
    -------
    wav : np.ndarray 
        The preprocessed waveform as a numpy array,
        ready to be used to compute the mel-filterbank energies.
    """
    #If waveform was given as a str path or a Path object instance, then needs to be loaded 
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    #Resample the wav if needed 
    if source_sr and source_sr != params_data.sr: 
        wav = librosa.resample(wav, source_sr, params_data.sr)

    #Normalize volume 
    if normalize_vol:
        wav = normalize_volume(wav, params_data.audio_norm_target_dbFS, increase_only=True)
    
    #Shorten long silences 
    if trim_silence:
        wav = trim_long_silences(wav)

    return wav

def normalize_volume(wav: np.ndarray, target_dbFS :int,
                      increase_only: bool =False, decrease_only: bool =False) -> np.ndarray: #Explained in the documentation
    """
    Normalizes the signals amplitud relative to how far bellow is from its peak amplitude. 
    
    Parameters
    ----------
    wav : np.array 
        The waveform that is getting normalized.
    target_dbFS : int
        Target decibel full-scale (dBFS).
    increase_only : bool, default = False 
        Determines wether only increase the amplitude of the waveform.
    decrease_only: bool, default = False
        Determines wether only decrease the amplitude of the waveform.

    Returns
    -------
    wav : np.ndarray 
        The normalized waveform.

    Raises
    ------
    ValueError
        If both 'increase_only' and 'decrease_only' are set to True.
    """
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    #Calculate the dBFS change that needs to be applied to the waveform to reach the target dBFS
    dBFS_change = target_dbFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20)) #Explained in the documentation 

def trim_long_silences(wav: np.ndarray) -> np.ndarray:
    """
    Ensures that the segments without voice in the waveform are no greater 
    than a threshold determined by the Voice Activity Detector (VAD) parameters
    defined in 'params_data.py' 

    Parameters
    ----------
    wav : np.ndarray
        the waveform as a numpy array of floats.
    Returns
    -------
    wav: np.ndarray
        the waveform with silenced segments trimmed -> trimmed_length <= original_length.
    """ 
    #voice detection window size 
    samples_per_window = (params_data.vad_window_length * params_data.sr) // 1000

    #Trim the end of the audio to have a multiple of the window size 
    wav = wav[:len(wav) - (len(wav) % samples_per_window)] #Removes the last "(len(wav) % samples_per_window)" values from the array

    #Convert the float wave form to 16-bit mono PCM audio bit depth 
    int16_max = (2 ** 15) - 1
    pcm_wave = pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16)) #Line explained in the documentation 

    #Perform voice activation detection 
    voice_flags = []
    vad = webrtcvad.Vad(mode=3) #Line explained in the documentation 
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window #Sliding the window
        #vad.is_speech() returns a binary value indicating if there was speech detected in the fame
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2: window_end * 2], sample_rate = params_data.sr))
    
    voice_flags = np.array(voice_flags)

    #Smooth the voice detection with a moving average #Explained in the documentation
    def moving_average(array, width): 
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2))) #Zero padding, Explained in the documentation
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, params_data.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    #Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(params_data.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]

def wav_to_mel_spectrogram(wav: np.ndarray) -> np.ndarray:
    """
    Computes the mel-spectrogram/mel-scaled spectrogram/mel frequency energies
    from a preprocessed audio waveform
    This is NOT log scaled mel spectrogram/log mel spectrogram

    Parameters
    ----------
    wav : np.ndarray
        The waveform as numpy array of floats.

    Returns
    -------
    frames : np.ndarray
        The output is a 2dimensional numpy array, a matrix, of shape '(n_frames, n_mels)', 
        the elements of this matrix are the mel filterbank energies of the waveform.
        TODO: For this implementation 'n_frames=hop_length=len(wav)=int(sr * window_step / 1000)'
    """
    frames = librosa.feature.melspectrogram(y=wav,
                                             sr=params_data.sr,
                                             n_fft=int(params_data.sr * params_data.window_length / 1000),
                                             hop_length=int(params_data.sr * params_data.window_step / 1000),
                                             n_mels = params_data.n_mels
                                            )
    return frames.astype(np.float32).T #Transposing so that the time axis becomes the first axis
