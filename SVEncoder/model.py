import numpy as np
import torch 
from torch import nn
from sklearn.metrics import roc_curve 
from scipy.interpolate import interp1d 
from scipy.optimize import brentq
from typing import Tuple 

import params_model
import params_data

'''
"GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION" - https://arxiv.org/pdf/1710.10467.pdf [1]
'''

class Encoder(nn.Module):
    """
    Speaker Encoder class.
    This class implements a speaker enconder as described in google's paper [1], 
    some modifications are made to tailor the encoder to this project's 
    goal, please refer to my bachelor's thesis .pdf for further details. 

    Attributes
    ----------
    loss_device : torch.device 
        The device (CPU or GPU/Cuda) on which the loss computations will be perfomed.
    loss_method : str 
        The method used for loss computation, either "softmax" as per eq.(6) from [1] or
        "contrast" as per eq.(7) from [1].
    lstm : nn.LSTM
        The LSTM network used for processing the input batch of mel frequency energies.
    linear : nn.Linear 
        The fully-connected layer applied to the output of the last LSTM layer.
    relu : nn.ReLU
        The ReLU activation layer applied to the output of the linear layer.
    similarity_weight : nn.Parameter 
        The parameter for scaling the similarity matrix.
    similarity_bias : nn.Parameter
        The parameter for biasing the similarity matrix. 

    Methods
    -------
    do_gradient_ops() -> None
        Performs operations on gradients to prevent exploding gradients during backpropagation.
    forward(utterances: torch.Tensor, hidden_init: torch.Tensor=None) -> torch.Tensor
        Computes the embedding of a batch of utterance mel frequency energies/spectrograms.
    similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor
        Computes the similarity matrix of the embeddings as described in section 2.1 of [1].
    loss_softmax(embeddings: torch.Tensor) -> Tuple[torch.Tensor, float]
        Computes the softmax loss as described in section 2.1 equation (6) of [1], 
        and the Equal Error Rate (EER).
    loss_contrast(embeddings: torch.Tensor) -> Tuple[torch.Tensor, float]
        Computes the contrast loss as described in section 2.1 equation (7) of [1], 
        and the Equal Error Rate (EER).
    """
    def __init__(self, device: torch.device, loss_device: torch.device, loss_method:str ="softmax"):
        super().__init__()
        self.loss_device = loss_device
        self.loss_method = loss_method

        #Network architecture definition
        #The input network pipeline of the End-to-End architecture is an LSTM network, as described in google's paper [1]
        self.lstm = nn.LSTM(input_size=params_data.n_mels,
                            hidden_size=params_model.model_hidden_size,
                            num_layers=params_model.model_num_layers,
                            batch_first=True
                            ).to(device)
        #A linear fully-connected layer is applied to the output of the last layer of the LSTM network, as described in [1]
        self.linear = nn.Linear(in_features=params_model.model_hidden_size,
                                out_features=params_model.model_embedding_size
                               ).to(device)
        #A Rectified Linear Unit ReLU layer is applied to the output of the fully-connected layer
        self.relu = nn.ReLU().to(device)

        #Initialization of the cosine similarity parameters,
        #the inital values are fixed
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)
    
        assert self.loss_method in ["softmax", "contrast"]
        if self.loss_method == "softmax":
            self.loss = self.loss_softmax
        if self.loss_method == "contrast":
            self.loss = self.loss_contrast

        #Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def do_gradient_ops(self) -> None: 
        """
        Performs scaling and clipping operations on gradients 
        to prevent exploding gradients during backpropagation.
        """
        #Gradient scale 
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01

        #Gradient clipping 
        nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=3, norm_type=2)
        
    def forward(self, utterances: torch.Tensor, hidden_init: torch.Tensor=None) -> torch.Tensor:
        """
        Computes the embedding of a batch of utterance mel frequency energies/spectrograms. 
            
        Parameters
        ----------
        utterances : torch.Tensor
            Batch of mel frequency energies of same duration as a tensor of shape 
            (batch_size=n_speakers * utterances_per_speaker, n_frames, n_mels).
        hidden_init : torch.Tensor 
            initial hidden state of the LSTM network as a tensor of shape
            '(num_layers=model_num_layers, batch_size=n_speakers * utterances_per_speaker, hidden_size=model_hidden_size)',
            will default to a tensor of zeros if None.

        Returns
        -------
        torch.Tensor 
            The embedding of the utterances as a tensor of shape 
            '(batch_size=n_speakers * utterances_per_speaker, model_embedding_size)'.
        """
        #Pass the input thourgh the LSTM layers and retreive all outputs, the final hidden state 
        # and the final cell state
        out, (hidden, cell) = self.lstm(utterances, hidden_init) 

        #Take only the hidden state of the last layer
        #Passing the output of last layer, i.e, the last hidden state, of the LSTM network through a fully-connected layer
        #And then the output of the fully-connected layer to a relu layer
        embeds_unnorm = self.relu(self.linear(hidden[-1]))      

        #L2 normalization of the embeddings
        #As per section 2.1 equation (4) of [1]
        embeddings = embeds_unnorm / (torch.norm(input=embeds_unnorm, dim=1, keepdim=True) + 1e-5)

        return embeddings 
        
    #****In the training loop a reshape is performed on the "embeddings" obtained from the output of "forward()",
    # the reshaping has the form:
    # (speakers_per_batch * utterances_per_speaker, model_embedding_size) -> (speakers_per_batch, utterances_per_speaker, model_embedding_size),
    # so, it changes from a matrix shape to a tensor shape.  

    def similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity matrix of the embeddings as described in section 2.1 of [1],
        First, the centroids are computed, both inclusive and exlusive, 
        as described in section 1.2 equation (1) and in section 2.1 equation (8) of [1],
        Inclusive centroids - 1 per speaker: One centroid per speaker, and each speaker's centroid include
        all their utterances.
        Exclusive centroids - 1 per utterances: One centroid per utterance, and each utterance's centroid 
        excludes its own embedding.  
        Then, the similarity matrix is computed as described in section 2.1 equation (9) of [1]

        Parameters
        ----------
        embeddings : torch.Tensor 
            A reshape from the utterances embeddings obtained from the "forward()" method 
            as a tensor of shape '(speakers_per_batch, utterances_per_speaker, model_embedding_size)'.
            The reason of the reshape is because is easier to make predictions, as explained in [1].

        Returns
        -------
        S_matrix : torch.Tensor
            The similarity matrix as a tensor of shape '(speakers_per_batch, utterances_per_speaker, speakers_per_batch)'.
        """
        speakers_per_batch, utterances_per_speaker = embeddings.shape[:2]

        #Centroids computation
        #Inclusive centroids - 1 per speaker. Cloning is needed for reverse differentiation.
        cents_inc = torch.mean(embeddings, dim=1, keepdim=True)
        cents_inc = cents_inc.clone() / (torch.norm(cents_inc, dim=2, keepdim=True) + 1e-5) #Normalized again to properly implement cosine similarity

        #Exclusive centroids - 1 per utterance
        cents_excl = (torch.sum(embeddings, dim=1, keepdim=True) - embeddings) 
        cents_excl /= (utterances_per_speaker - 1)
        cents_excl = cents_excl.clone() / (torch.norm(cents_excl, dim=2, keepdim=True) + 1e-5)  #Normalized again to properly implement cosine similarity

        #Similarity matrix computation
        #The cosine similarity of two 2-normed vectors is their dot product
        #Vectorizing the dot product for faster performance
        S_matrix = torch.zeros(size=(speakers_per_batch, utterances_per_speaker, speakers_per_batch) ).to(self.loss_device)
        mask_matrix = 1 - np.eye(N=speakers_per_batch, dtype=int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            S_matrix[mask, :, j] = (embeddings[mask] * cents_inc[j]).sum(dim=2) #Selects embeddings of all other speakers and computes their dot product with the inclusive centroid for the current j-th speaker
            S_matrix[j, :, j] = (embeddings[j] * cents_excl[j]).sum(dim=1)  #Selects embeddings of the j-th speaker and computes their dot product with the exclusive centroid for the current j-th speaker

        S_matrix = S_matrix * self.similarity_weight + self.similarity_bias
        return S_matrix
            
    def loss_softmax(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Computes the softmax loss as described in section 2.1 equation (6) of [1], 
        and the Equal Error Rate (EER).

        Parameters
        ----------
        embeddings : torch.Tensor 
            A reshape from the utterances embeddings obtained from the "forward()" method 
            as a tensor of shape '(speakers_per_batch, utterances_per_speaker, model_embedding_size)'.
            
        Returns
        -------
        (loss, eer) : Tuple[torch.Tensor, float]
            The loss and the EER for the batch of embeddings.
        """
        #****Why is 'speakers_per_batch' and 'utterances_per_speaker' reassigned?, their values are originally
        # defined at params_model, do they change at some point when computing the embeddings????
        speakers_per_batch, utterances_per_speaker = embeddings.shape[:2]

        #SoftmaxLoss 
        S_matrix = self.similarity_matrix(embeddings=embeddings)
        S_matrix = S_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch)) #Check documentation
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker) #Check documentation
        target = torch.from_numpy(ground_truth).long().to(self.loss_device) 
        loss = self.loss_fn(S_matrix, target)

        #Equal Error Rate EER , is not part of the training process, therefore it needs no gradient computation
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth]) #shape (speakers_per_batch * utterances_per_speaker, speakers_per_batch)
            predictions = S_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), predictions.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        return loss, eer 

    def loss_contrast(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Computes the contrast loss as described in section 2.1 equation (7) of [1], 
        and the Equal Error Rate (EER).

        Parameters
        ----------
        embeddings : torch.Tensor
            A reshape from the utterances embeddings obtained from the "forward()" method 
            as a tensor of shape '(speakers_per_batch, utterances_per_speaker, model_embedding_size)'.

        Returns
        -------
        (loss, eer) : Tuple[torch.Tensor, float]
            The loss and the EER for the batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeddings.shape[:2]
        S_matrix = self.similarity_matrix(embeddings=embeddings)
        
        pos_mask = list(range(speakers_per_batch))
        S_pos = S_matrix[pos_mask, :, pos_mask]
        S_neg = torch.max(S_matrix - torch.eye(speakers_per_batch).unsqueeze(1)*1e9, dim=2)[0]
        #L = 1 - torch.sigmoid(S_pos) + torch.sigmoid(S_neg)
        loss = torch.mean(1 - torch.sigmoid(S_pos) + torch.sigmoid(S_neg)).to(self.loss_device)
        
        S_matrix = S_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch)) 
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)  
        
        #Equal Error Rate EER , is not part of the training process, therefore it needs no gradient computation
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth]) #shape (speakers_per_batch * utterances_per_speaker, speakers_per_batch)
            predictions = S_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), predictions.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer