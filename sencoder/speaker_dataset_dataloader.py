from pathlib import Path
from typing import  List, Iterable

from torch.utils.data import Dataset, DataLoader

import params_data
from speaker import Speaker
from randomcycler import RandomCycler
from speaker_batch import SpeakerBatch

'''
"GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION" - https://arxiv.org/pdf/1710.10467.pdf [1]
'''

class SpeakerDataset(Dataset):
    """
    This class is designed to be used as a PyTorch 'Dataset' and represents a collection of speakers 
    whose data are preprocessed mel frequency energies of their audio frames. It allows 
    access and iteration over the speakers' data in a constrained random order.
    The main funcionality of this class is to enable a constrained random access to the 
    'Speaker' instances in the dataset. It achieves this by shuffling the 'Speaker' instances
    and providing a method to repeatedly sample 'Speaker' instances in a constrained 
    random manner.

        Attributes
        ----------
        root : Path
            A 'pathlib.Path' instance of the directory where all the speakers' subdirectories 
            are located, each subdirectory contains the .npy files of the preprocessed
            mel frequency energies of the frames corresponding to the speaker's utterances.
        speakers : List[Speaker]
            A list of 'Speaker' instances, with each 'Speaker' instance corresponding to a speaker 
            from the dataset and containing their corresponding preprocessed data
        speaker_cycler : RandomCycler
            A 'RandomCycler' instance that allows constrained random access to the 'Speaker' instances
            that are located at 'speakers'. It provides a way to shuffle the 'speakers' list and 
            repeatedly sample 'Speaker' instances in a way that guarantees each instance is sampled 
            a certain number of times while preserving randomness.
        """
    def __init__(self, preprocessed_dataset_root: Path):
        """

        Parameters
        ----------
        preprocessed_dataset_root : Path
            A 'pathlib.Path' instance of the location where all the speakers' subdirectories 
            are locared, each subdirectory contains the .npy files of the preprocessed
            mel frequency energies of the frames corresponding to the speaker's utterances.
        """
        self.root = preprocessed_dataset_root
        speaker_dirs = [dir for dir in self.root.glob("*") if dir.is_dir()] #A list containig all the path to the directories located as 'root'
        if len(speaker_dirs) == 0:
            raise Exception("No speaker found. Passed path must be pointing to the directory containing all preprocessed speaker directories") 
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs] 
        self.speaker_cycler = RandomCycler(self.speakers) #To randomize speakers access

    def __len__(self) -> int:
        """
        To override python's built-in 'len()' method.
        Is used to specify the number of items in the dataset, which serves as the total
        number of iterations the dataset can produce. In this case a large number 1e10 is
        returned to allow for "infinite" iterations. Even though the dataset does not 
        have 1e10 speakers. The purpose of this is for The data loader to loop over 
        the dataset indifinitely, allowing the model to see a different subset
        of the data each time it loops. This is often used when training a model
        on large datasets where it is not feasible to load the entire dataset into memory at once. 

        Returns
        -------
        int 
            The return value of this overriden method is not intended to be applied anywhere
            its purpose is for The data loader to loop over the dataset indifinitely,
            allowing the model to see a different subset of the data each time it loops.
            The returned value does not represent the real number of speakers from the dataset.
        """ 
        return int(1e10)
        
    def __getitem__(self, index) -> Speaker:
        """
        Overrides the bahviour of Python's built-in indexing operator '[]' when applied
        to an instance of the class, that is, 'class_instance[index]'.

        Returns
        -------
        Speaker
            A 'Speaker' instance from the 'speakers' list, sampled by 'speaker_cycler'.
        """
        return next(self.speaker_cycler) 

    def get_logs(self) -> str:
        """
        Retrieves the logs from the root directory, reading all '.txt' files and
        concatenating their contents.

        Returns
        -------
        str
            A string containing the concatenated contents of all the log files 
            found in the root directory.
        """
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string 
    

class SpeakerDataLoader(DataLoader):
    """
    A custom dataloader for the 'SpeakerDataset' class. This class extends the functionality 
    of PyTorch's utility class 'DataLoader' to work with the custom class 'SpeakerDataset',
    it provides a way to load batches of data from the 'SpeakerDataset' instance. 

    Attributes
    ----------
    spakers_per_batch : int 
        The number of speakers to sample when creating a batch. This values is 
        assigned to the 'batch_size' parameter of Pythorch's 'DataLoader' utility
        class.
    utterances_per_speaker : int 
        The number of utterances to sample per sampled speaker when creating a batch. 

    Methods
    -------
    collate(speakers: List[Speaker])

    """
    def __init__(self, dataset: SpeakerDataset, speakers_per_batch: int, utterances_per_speaker: int,
                sampler: Iterable = None, batch_sampler = None, num_workers: int = 0,
                pin_memory: bool =False, timeout: int=0, workers_init_fn=None):
        """
        Initializes the 'SpeakerDataLoader' with the given parameters and passes them 
        to the PyTorch's base 'DataLoader' constructor. 
        'shuffle' is set to 'False' so that the dataset is not shuffled by PyTorch's 'DataLoader' 
        before each epoch during training, this is done because the 'SpeakerDataset' class implements
        a custom shuffler designed to provide constrained random access to 'Speaker' instances
        of the dataset. Meaning that, although the order of speakers is random, there are constraints
        on how many times each speaker should be sampled and the order in which they appear.
        Such constrains are defined in the custom class 'RandomCycler'.

        Parameters
        ----------
        dataset : SpeakerDataset
            A 'SpeakerDataset' instance that stores the collection of speakers' 'Speaker'
            instances, from where the data will be loaded. 
        spakers_per_batch : int 
            The number of speakers to sample when creating a batch. This values is 
            assigned to the 'batch_size' parameter of Pythorch's 'DataLoader' utility
            class.
        utterances_per_speaker : int 
            The number of utterances to sample per sampled speaker when creating a batch.
        The rest of parameters are from Pythorch's 'DataLoader' utility class.
        """    
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset,
            batch_size=speakers_per_batch,
            shuffle=False,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory,
            drop_last=False,
            timeout=timeout,
            worker_init_fn=workers_init_fn
        )
    
    def collate(self, speakers: List[Speaker]) -> SpeakerBatch:
        """
        Custom collate function used by the 'DataLoader' to create batches of data.

        Parameters
        ----------
        speakers : List[Speaker]
            A list of 'Speaker' instances to create a batch from.

        Returns
        -------
        SpeakerBatch
            A 'SpeakerBatch' that represents a batch of data from the 'SpeakerDataset'.
            It organizes the mel filterbank energies in a format that has the shape of
            the similarity matrix as described by google's [1] paper in eq.(9) and fig.1 .
        """
        return SpeakerBatch(speakers, self.utterances_per_speaker, params_data.partials_n_frames)
