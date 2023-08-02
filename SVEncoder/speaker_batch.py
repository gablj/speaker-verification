import numpy as np
from typing import List

from speaker import Speaker

'''
"GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION" - https://arxiv.org/pdf/1710.10467.pdf [1]
'''

class SpeakerBatch:
   """
   A class to generate speakers' utterances data batches, the batches
   have the shape of the similarity matrix as described by google's [1] paper
   in eq.(9) and fig.1 .

   Attributes
   ----------
   speakers : List[Speaker]
      A list of 'Speaker' instances, with each 'Speaker' instance corresponding to a speaker 
      from the dataset and containing their corresponding preprocessed data.
   partial_utterances : Dict
      A dictionary that stores pairs each corresponding 'Speaker' instance from the 'speakers' list to
      a list containing 'utterances_per_speaker' utterances from the corresponding 'Speaker' instance.
   data : np.ndarray
      3dimensional np.ndarray, i.e, a tensor (in the mathematical sense not in Pytorch's object sense, that is,
      a matrix whose elements are submatrices) of shape '(speaker_per_batch * utterances_per_speaker, n_frames, n_mels)',
      conceptually speaking, the resulting ndarray has 'speakers_per_batch * utterances_per_speaker' submatrices,
      each submatrix has 'n_frames' rows and 'n_mels' columns, i.e, 'data.shape[0] = speakers_per_batch * utterances_per_speaker', 
      'data.shape[1:] = (n_frames, n_mels)'. The batch is constructed with these dimensions as described in
      google's paper [1].

      Example: 64 speakers_per_batch, 10 utterances_per_speaker, each utterance represented through the
      mel spectrogram matrix of 160 frames (rows), each frame of 40 mel coefficients (columns) yields
      the batch dimentions (640, 160, 40).
   """
   def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
      """
      Parameters
      ----------
      speakers : List[Speaker]
         A list of 'Speaker' instances, with each 'Speaker' instance corresponding to a speaker 
         from the dataset and containing their corresponding preprocessed data.
      utterances_per_speaker: int
         The number of utterances to sample per sampled speaker when creating a batch.
      n_frames : int
         The number of frames per utterance.
      """
      self.speakers = speakers 
      self.partial_utterances = {speaker: speaker.random_partial_segment(utterances_per_speaker, n_frames) for speaker in speakers}
      self.data = np.array([frames for speaker in speakers for _, frames, _ in self.partial_utterances[speaker]]) 