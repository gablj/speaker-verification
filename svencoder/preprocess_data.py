"""
This module defines a class, 'PreprocessData', to execute the preprocessing
of the audiofiles from the 'LibriSpeech ASR corpus' 'other-500',
dataset source: http://www.openslr.org/12, which consists of audio samples
featuring read english speech. The "train-other-500" subset has a size of 30 GB
and a total duration of ∼ 500 hours of recorded audio samples, from 1, 166 speakers,
comprising 564 female speakers and 602 male speakers. Each speaker’s samples
add to a total duration of ∼ 30 minutes. Most of the speech is sampled at 16𝑘𝐻𝑧.
It is worth to point that, since the audio samples from the dataset is sourced from audio
books, the tone and style of speech can differ significantly between utterances from the same
speaker.
The audiofiles are processed to obtain their mel-filerbank energies as numpy arrays of floats
in .npy files. These mel-filterbank energies are the features fed to the deep learning model. 
"""

from pathlib import Path 
from typing import List
from functools import partial 
from multiprocessing import Pool
from tqdm import tqdm 

import numpy as np 

from svencoder import params_data
from svencoder import preprocess_functions
from svencoder.data_log import DataLog

_AUDIO_EXTENSIONS = ("wav", "flac", "m4a", "mp3") 

class PreprocessData:
    """
    This class defines a set of methods, whose pourpose is to execute the preprocessing
    of the audiofiles from the 'LibriSpeech ASR corpus' 'other-500',
    dataset source: http://www.openslr.org/12.
    The audiofiles are processed to obtain their mel-filerbank energies as numpy arrays of floats
    in .npy files. These mel-filterbank energies are the features fed to the deep learning model. 
    Furthermore, an instance of the class 'DataLog' is initialized to create
    a .txt file with metadata of the preprocessing of the dataset. 

    The resulting directory has the following scheme: 
        .- out_path/
            |
                .- LibrisPeech_train-other-500_XXXX/
                |    |
                    _sources.txt
                    sample_id_xxxx.npy
                    sample_id_xxxy.npy
                    ...
                .- LibrisPeech_train-other-500_YYYY/
                    |
                    ...
                ...
                .- log_LibriSpeech_train-other-500.txt

    Attributes
    ----------
    root_path : Path
        A 'pathlib.Path' instance of the directory where the dataset/all speakers directories
        are stored. 
    out_path : Path 
        A 'pathlib.Path' instance of the output directory, where the preprocessed output files
        will get stored.

    Methods
    -------
    preprocess_data(skip_existing: bool = False)
        Sets in execution the data preprocessing and data log. 

    """

    def __init__(self, root_path: Path, out_path: Path):
        """
        Attributes
        ----------
        root_path : Path
            A 'pathlib.Path' instance of the directory where the dataset/all speakers directories
            are stored. 
        out_path : Path 
            A 'pathlib.Path' instance of the output directory, where the preprocessed output files
            will get stored.
        """
        self.root_path = root_path
        self.out_path = out_path

    def __preprocess_speaker(self, speaker_dir: Path, skip_existing: bool) -> List:
        """
        Private method (not intended to be accessed outside the class or by users)
        that creates an output directory for a speaker and calls the corresponding functions
        from 'PreprocessFuncions.py' to compute and store the mel-filterbank energies
        as numpy arrays in the form of 'npy' files for each of the speaker's audio recordings. 
        It also computes the duration of each of the speakers audio files.
        
        Parameters
        ----------
        speaker_dir : Path 
            A 'pathlib.Path' instance of the directory where the speaker's audio file recordings
            are stored.
        skip_existing : bool 
            If 'False' will re-compute already preprocessed audiofiles, else will skip
            existing ones.
        
        Returns
        -------
        audio_durs : List 
            A list containing the duration in seconds of each of the speaker's audio files.

        """
        #Names the speaker as "LibrisPeech_train-other-500_XXXX"
        #where the last "XXXX" represents the directory number
        speaker_name = "_".join(speaker_dir.relative_to(self.root_path).parts)
            
        #Creates an output directory for "speaker_name", also
        #creates a txt file containing a reference to each source file
        speaker_out_dir = self.out_path.joinpath(speaker_name)
        speaker_out_dir.mkdir(exist_ok=True) #Creates directory, "exist_ok=True" ignores the error "FileExistsError" 
        #To store the npy file name and source audiofile location as 'frames.npy, sourc_path'
        sources_fpath = speaker_out_dir.joinpath("_sources.txt")
            
        #In case that the preprocessing was interrupted, 
        #Check if there are already source files
        if sources_fpath.exists():
            try:     #Try the next code block to check if there's an error
                with sources_fpath.open("r") as sources_file:
                        existing_fnames = {line.split(",")[0] for line in sources_file} #This is a set 
            except:  #Handles the error
                existing_fnames = {}
            #else:   #Execute when there's no error
            #    print("No errors")
        else:  
            existing_fnames = {}
            
        #Gather all audio files recursively 
        sources_file = sources_fpath.open("a" if skip_existing else "w") #The flag "a" opens the file for writing and if there's something written it appends the new content to it, the flag "w" does the same but if there's already content it eareases it
        audio_durs = []
        for extension in _AUDIO_EXTENSIONS:
            for in_fpath in speaker_dir.glob("**/*.%s" % extension): #Look for on the directory and all subdirectories for paths of files ending in "extension"
                    
                #Names the output file
                out_fname = "_".join(in_fpath.relative_to(speaker_dir).parts) 
                out_fname = out_fname.replace(".%s" % extension, ".npy")    #Changing extension to "numpy", this is NOT a file conversion from flac to npy, is just renaming the file where the numpy array will be saved
                    
                #Check if the target outputfile already exists
                if skip_existing and out_fname in existing_fnames: #if "skip_existing = False", it proceeds to re-write it in the bellow lines
                    continue 
                    
                #Load and Preprocess the waveform 
                wav = preprocess_functions.preprocess_wav(in_fpath)
                if len(wav) == 0:
                    continue
                    
                #Compute the mel spectrogram/mel frequency energies 
                frames = preprocess_functions.wav_to_mel_spectrogram(wav)
                if len(frames) < params_data.partials_n_frames: #Discards the spectrogram if it doesn't have at least "partials_n_frames" frames
                    continue
                        
                out_fpath = speaker_out_dir.joinpath(out_fname)
                np.save(out_fpath, frames)
                sources_file.write("%s, %s \n" %(out_fname, in_fpath))
                audio_durs.append(len(wav) / params_data.sr)
                    
        sources_file.close()
            
        return audio_durs

    def _preprocess_speaker_wrapper(self, speaker_dir: Path, skip_existing: bool) -> List:
        """
        This 'wrapper' method is needed since 'functools.partial()' doesn't 
        have access to the private method '__preprocess_speaker()'.
        Protected method to wrap the '__preprocess_speaker()' call 
        for parallel processing.

        Parameters
        ----------
        speaker_dir : Path 
            A 'pathlib.Path' instance of the directory where the speaker's audio file recordings
            are stored.
        skip_existing : bool 
            If 'False' will re-compute already preprocessed audiofiles, else will skip
            existing ones.

        Returns
        -------
        audio_durs : List 
            A list containing the duration in seconds of each of the speaker's audio files.
            As per the implementation of '__preprocess_speaker()'.
        """
        return self.__preprocess_speaker(speaker_dir, skip_existing)

    def __preprocess_speaker_dirs(self, speaker_dirs: List[Path], data_name: str,
                                skip_existing: bool, datalog: DataLog) -> None: 
        """
        Private method (not intended to be accessed outside the class or by users) that calls
        '_preprocess_speaker()' to each of the speakers directories from the main database directory.
        This is perfomed in a parallelized way.
        
        Parameters
        ---------
        speaker_dirs : List[Path] 
            A list of 'pathlib.Path' instances of the speakers directories.
        data_name : str 
            The name of the directory/dataset where the audio files are stored, i.e., not the complete path
            but the directory/folder name.
        skip_existing : bool 
            If 'False' will re-compute previously preprocessed audiofiles, else will skip
            existing ones.
        datalog : Datalog 
            A 'Datalog' instance that will be used to create the datalog 
        """
        print(f"from __preprocess_speaker_dirs: Preprocessing data for {data_name} containing {len(speaker_dirs)} speakers" )
        
        #Revise 
        work_fn = partial(self._preprocess_speaker_wrapper, skip_existing=skip_existing)
        with Pool(processes=4) as pool: #For parallelization
            tasks = pool.imap(work_fn, speaker_dirs) #To apply function "_preprocess_speaker_wrapper()" to each of the paths of "speaker_dirs"
            for sample_durs in tqdm(tasks, data_name, len(speaker_dirs), unit="speakers"):
                for sample_dur in sample_durs:
                    datalog.add_sample(duration=sample_dur)
        
        datalog.finalize()
        print("from __preprocess_speaker_dirs:Finished")
    
    def preprocess_data(self, skip_existing: bool = False) -> None:
        """
        This is the only method that is intended to be called by the user to initiate the
        data preprocesing. This is the function that initiates the 'DataLog' instance and
        sets in execution the datapreprocesing.
        
        Parameters
        ----------
        skip_existing : bool, dafault = False 
            If 'False' will re-compute already preprocessed audiofiles,
            else will skip existing ones.
        """
        if skip_existing:
            print("****\"skip_existing\" set to \"True\" - any existing file will not be re-processed/recomputed, ")
            print("****if the parameters have been changed, set \"skip_existing\" to \"False\" so that the files are re-processed/recomputed with the updated parameters \n")
        else:
            print("****\"skip_existing\" set to \"False\" - any existing file will be re-processed/recomputed, ")
            print("****if the parameters have not been changed, set \"skip_existing\" to \"True\" so that the files are not re-processed/recomputed \n")

        if not self.root_path.exists(): #Check wether the path points to an existing directory 
            print(f"No directory {self.root_path} found")
            return
        data_name = self.root_path.as_posix().split('/')[-1] #'as_posix()' to get the path as a string
        datalog = DataLog(self.out_path, data_name)
            
        speakers_dirs = list(self.root_path.glob("*")) #To get all directories paths
        self.__preprocess_speaker_dirs(speakers_dirs, data_name, skip_existing, datalog) 
