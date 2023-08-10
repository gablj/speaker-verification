from utterance import Utterance
from randomcycler import RandomCycler

import numpy as np
from pathlib import Path
from typing import List, Tuple #For type hinting 

class Speaker: 
    """
    A class representing a speaker, this class acts as a container for 
    multiple 'Utterance' instances. When a new 'Speaker' object is created 
    the 'utterances' attribute is populated with a list of 'Utterance' instances, 
    each representing a specific utterance of that speaker. The relation between
    the 'Speaker' and 'Utterance' classes is a "Composition" relationship, where 
    the 'Speaker' class "has-a" collection of 'Utterance' instances. This hierarchical
    and structured design helps in organizing and managing data related to speakers
    and their respective utterances in a more modular and flexible way.
    This class also has a 'RandomCycler' instance whose purpose is to provide 
    contrained random access to the 'Utterance' instances contained in the 'Speaker' instance.
    
    Attributes
    ----------
    root_dir : Path
        A 'pathlib.Path' instance of the directory where all preprocessed '.npy'
        mel frequency energies of the speaker are located. 
    speaker_id : str 
        The number id of the corresponding speaker.
    utterances : List, default = None
        A list containing 'Utterance' instances each corresponding to a specific utterance of the speaker,
        is initialized to 'None', and populated when the method 'random_partial_segment()' is called.
    utterance_cyler : RandomCycler
        A 'RandomCycler' instance that allows constrained random access to the 'Utterance' instances
        that are located at 'utterances'. It provides a way to shuffle the 'utterances' list and 
        repeatedly sample 'Utterance' instances in a way that guarantees each instance is sampled 
        a certain number of times while preserving randomness.

    Methods
    -------
    random_partial_segment(count: int, n_frames: int)
        Samples a batch of 'count' unique partial utterances in a way that all utterances come up
        at least once every two cyles and in a random order every time.
    """
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir 
        self.speaker_id = root_dir.name
        self.utterances = None 
        self.utterance_cycler = None
    
    def __load_utterances(self) -> None:
        """
        Loads the file from path 'frames_fpath' that has a numpy array containing mel frequency energies 
        of the frames
        computes and assigns:
        sources : Dict 
            A dictionary with key:value pairs as npy_file_name:path_of_sound_file where
            'npy_file_name' are the mel frequency energies from the frames of the waveform 
            from located at 'path_of_sound_file'
        utterances : List 
            A list containing 'Utterance' instances each corresponding to a specific utterance of the speaker,
            is initialized to 'None', and populated when the method 'random_partial_segment()' is called.
        utterance_cyler : RandomCycler
            A 'RandomCycler' instance that allows constrained random access to the 'Utterance' instances
            that are located at 'utterances'. It provides a way to shuffle the 'utterances' list and 
            repeatedly sample 'Utterance' instances in a way that guarantees each instance is sampled 
            a certain number of times while preserving randomness.
        """ 
        with self.root_dir.joinpath("_frames.txt").open("r") as sources_file:
            #a list of tuples where each element is a list of the form [npy_file_name, target_speaker]
            sources = [line.split(",") for line in sources_file]
        sources ={frames_fname: target for frames_fname, target in sources}
        self.utterances = [Utterance(frames_fpath=self.root_dir.joinpath(frames_fname), target_speaker=int(target) ) for frames_fname, target in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances) 

    def random_partial_segment(self, count: int, n_frames: int) -> List[Tuple[Utterance, np.ndarray, Tuple[int, int]]]:
        """ 
        Samples a batch of 'count' unique partial utterances in a way that all utterances come up
        at least once every two cyles and in a random order every time.

        Parameters
        ----------
        count : int 
            The number of partial utterances to sample from the set of total utterances from the speaker,
            utterances are not repeated if count <= total utterances available. 
        n_frames : int 
            The number of frames in each of the partial utterances.
        
        Returns
        -------
        List[Tuple[Utterance, np.ndarray, Tuple[int, int]]] 
            A list of tuples that have the form: '(utterance, frames, range)' where 'utterance'
            is the corresponding 'Utterance' instance, 'frames' is a numpy array with the mel-frequency energies of
            the corresponding partial utterance and 'range' is tuple with the indexed of the start and end of the 
            partial utterance frames relative to the complete utterance.
        """
        if self.utterances is None:
            self.__load_utterances()

        utterances = self.utterance_cycler.sample(count)
        UFR = [(utter,) + utter.random_partial_segment(n_frames) for utter in utterances]
        return UFR

    def check_target(self) -> bool:
        """ 
        A method added just to check if the target of each of the contained 'Utterance'
        instances inside the 'Speaker' instance is the same as the 'Speaker' instance id.

        Returns
        -------
        bool
            Returns 'True' if all 'Utterance' instances contained have the same 
            target speaker id as the 'Speaker' instance.
        """
        if self.utterances is None:
            print("No utterances loaded ")
            return False
        
        return all(utter.target_speaker == self.speaker_id for utter in self.utterances)
    
