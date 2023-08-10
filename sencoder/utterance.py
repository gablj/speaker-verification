import numpy as np
from pathlib import Path
from typing import Tuple


class Utterance: 
    """"
    A class representing an utterance through the mel-frequency energies from an audifile.
    
    Attributes
    ----------
    frames_fpath : Path 
        The pathlib.Path of the npy file containing mel frequency energies of the frames
        from the sound file.
    target_speaker : int
        The id of the target speaker associated with the utterance.

    Methods
    -------
    get_frames()
        Loads the npy file, containing the frames' mel frequency energies, located
        at 'frames_path'.
    random_partial_segment(n_frames: int)
        Returns a random segment of 'n_frames' frames from the utterance.

    """
    def __init__(self, frames_fpath: Path, target_speaker: int):
        """
        Parameters
        ----------
        frames_fpath : Path 
            The pathlib.Path of the npy file containing mel frequency energies of the frames
            from the sound file.
        target_speaker : int
            The id of the target speaker associated with the utterance.
        """
        self.frames_fpath = frames_fpath
        self.target_speaker = target_speaker

    def get_frames(self) -> np.ndarray:
        """
        Loads the npy file, containing the frames' mel frequency energies, located
        at 'frames_path'.
        
        Returns
        -------
        np.ndarray
            A numpy array of floats containing the mel frequency energies of 
            the utterance's frames.
        """   
        return np.load(self.frames_fpath)
    
    def random_partial_segment(self, n_frames: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """ 
        Returns a random segment of 'n_frames' consecutive frames from the utterance.
        
        Parameters
        ----------
        n_frames : int
            The number of frames of the partial utterance segment.

        Returns
        -------
        Tuple[np.ndarray, Tuple[int, int]]
            A tuple of two elements: the first element is a numpy array with the partial
            utterance frames and the second element is a two-element tuple with the indexes of the
            start (int) and end (int) of the partial utterance frames relative to the complete utterance.
        """
        frames = self.get_frames()
        if frames.shape[0] == n_frames: 
            start = 0
        else: #Picks a random start point 
            start = np.random.randint(0, frames.shape[0] - n_frames) #Arguments: lower and upper limit for the random value
        end = start + n_frames 
        return frames[start:end], (start, end)

