## This Repository is Under Construction
Not all the source code has been uploaded, I'm currently working on modifying some aspects of the source code 
to use it in an upcoming project. Although I've already defended my thesis (and graduated), the modifications aren't meant to
improve the results obtained on my research, but rather to improve some minor portable features of the code.

The inference module will get uploaded soon. 

## Description
A repository for the source code of the end-to-end pipeline used in my Bachelor's Thesis: "Deep Learning Based End-to-End Text-Independent Speaker Verification".

This is a modified implementation of an end-to-end architecture described in google's paper "Generalized End-to-End Loss for Speaker Verification" - https://arxiv.org/pdf/1710.10467.pdf [1] . 

The model was trained using the audiofiles from the 'LibriSpeech ASR corpus' 'other-500', dataset source: http://www.openslr.org/12, which consists of audio samples
featuring read english speech. The "train-other-500" subset has a size of 30 GB and a total duration of ∼ 500 hours of recorded audio samples, from 1, 166 speakers,
comprising 564 female speakers and 602 male speakers. Each speaker’s samples add to a total duration of ∼ 30 minutes. Most of the speech is sampled at 16𝑘𝐻𝑧.

It is worth to point that, since the audio samples from the dataset is sourced from audio books, the tone and style of speech can differ significantly between utterances from the samespeaker.

The audiofiles are processed to obtain their mel-filerbank energies as numpy arrays of floats
in .npy files. These mel-filterbank energies are the features fed to the deep learning model.

## About the documentation
"Documentation is a love letter that you write to your future self." - a very wise man.

This project intends to follow PEP 8 style guide for python code: https://peps.python.org/pep-0008/ .

The documentation style is mostly the same as google's python style guide: https://google.github.io/styleguide/pyguide.html .

I've put extra effort into the documentation, since the speech processing part can be a bit tricky, in such a way that with little effort from the user, the relation and purpose between classes can be easily understood. 

# About the trained model results
The model ("encoder.pt" on /trained_models) was trained for $150 K$ epochs, the accuracy score, loss and equal error rate were tracked for both the training and testing sets throughout the training process. In addition to this, the use of the Uniform Manifold Approximation and Projection (UMAP) technique was incorporated, which is a dimensionality reduction technique that aims to capture the underlying structure and relationships in the data by mapping it to a lower-dimensional space. UMAP was implemented to periodically (every $100$ epochs) project a batch of $64$-dimensional embeddings into two-dimensional space to monitor how the model clusters speakers and to further contextualize the results, since, as the model's training and test losses decrease, it is expected that this would be reflected in the embedding space, as suggested by the GE2E loss [1]. Meaning that, is anticipated the formation of tight clusters of utterance embeddings from the same speaker as the training evolves, it is also expected to find a certain degree of separation between different clusters. The projections through UMAP provides a contextual understanding of the model's performance. We present some of these projections, for different epochs, where this behavior is presented:

![alt text](https://github.com/gablj/speaker-verification/blob/main/images/umap_projections.png)

As can be observed in the figure above, the initial $100$ steps show the training embedding space as highly unclustered. The embeddings corresponding to the same speaker are scattered and mixed with others. This pattern continued for roughly the first $1,000$ epochs from which after clusters start to become noticeable. By epoch $10,000$, embeddings of the same speaker are closer, but intersections with embeddings from different speakers are still very prominent.  At epoch $50, 000$ most of the same-speaker embeddings have significantly move closer to each other, although some remain distant or intersect with different clusters. Step $100,000$ reveals that tight clusters have been already formed, however, some of these clusters remain close to each other. At this point, the model learned to produce same-speaker embeddings that are close to each other, however, I continue the training process, allowing the model to further optimize the distance between different speakers clusters. Finally, by epoch $150, 000$ the embedding space projection shows that the distance between clusters has significantly increased.

I halted the training at $150 K$ steps since the model had converged by this point. The learning curves had flattened, showing no significant improvements. The loss values for the training set were already very low, fluctuating around $0.05$. The training accuracy value stabilized at approximately $0.97$ and the equal error rate remained bellow $1 \%$.
All of these are indicators that the model has converged. To continue the training further would yield no significant improvements.
Bellow, we present the evolution of the training curves and the values obtained from the training at epoch $150, 000$.

![alt text](https://github.com/gablj/speaker-verification/blob/main/images/loss_plot.png)
![alt text](https://github.com/gablj/speaker-verification/blob/main/images/eer_plot.png)
![alt text](https://github.com/gablj/speaker-verification/blob/main/images/accuracy_plot.png)
<img src="https://github.com/gablj/speaker-verification/blob/main/images/accuracy_plot.png" width="900" height="100">


Training Values at epoch $150, 000$.
| Train Loss | Train Accuracy | Train EER |
|------------|----------------|-----------|
| 0.048      | 0.987          | 0.5 \%    |

| Test Loss | Test Accuracy   | Test EER |
|------------|----------------|-----------|
| 0.725      | 0.793          |  5.7 \%    |

  
