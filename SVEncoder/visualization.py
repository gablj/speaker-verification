"""
This module provides a class for generating and updating visualizations using
Visdom https://github.com/fossasia/visdom, for monitoring and analyzing 
real-time training and testing data progress.
Visdom facilitates the visualization of (remote) data with an emphasis on supporting scientific experimentation,
it supports Torch and Numpy. 
"""

from datetime import datetime 
from time import perf_counter as timer
from pathlib import Path
from typing import Dict, Union 

import numpy as np
import matplotlib.pyplot as plt 
import umap     #https://github.com/lmcinnes/umap
import visdom   #https://github.com/fossasia/visdom

from speaker_dataset_dataloader import SpeakerDataset
import params_data
import params_model

#Colormap for the UMAP two-dimensional embedding projections.
colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=float) / 255

class Visualizations:
    """
    This class facilitates the creation and update of a Visdom environment
    for monitoring and analyzing real-time training and testing progress.

    Attributes
    ----------
    env_name : str, default=None
        The name of the Visdom environment.
    update_every : int, default=10
        The frequency (in number of steps/epochs) of updating the visualizations,
        this parameter also defines the interval at which the loss, accuracy and eer will
        be averaged.
    server : str, default="http://localhost"
        The Visdom server address. 
    disabled : bool, default=False
        Whether to disable the use of Visdom, and consequently, the visualizations.
    last_update_timestamp : float 
        Timestamp of the last visualizations update.
    step_times : List[float]
        A list that keeps track the duration of each training epoch within the 'update_every',
        is used to calculate the mean and std of training loops durations over the 'update_every' interval,
        helping with performance analysis and bottleneck identification during training.
        This list is cleared and repopulated at the end of each 'update_every' interval of epochs. 
    losses : List[float]
        A list that stores training losses. It accumulates losses over every 'update_every' interval of epochs,
        calculates their mean. The list is cleared and repopulated at the end of each 'update_every' interval of epochs.
    eers : List[float] 
        A list that stores training equal error rates (eers). It accumulates eers over every 'update_every' interval of epochs,
        calculates their mean. The list is cleared and repopulated at the end of each 'update_every' interval of epochs.
    accuracies : List[float]
        A list that stores training accuracies. It accumulates accuracies over every 'update_every' interval of epochs,
        calculates their mean. The list is cleared and repopulated at the end of each 'update_every' interval of epochs.
    test_losses : List[float] 
        A list that stores testing losses. It accumulates losses over every 'update_every' interval of epochs,
        calculates their mean. The list is cleared and repopulated at the end of each 'update_every' interval of epochs.
    test_eers : List[float]
        A list that stores testing equal error rates (eers). It accumulates eers over every 'update_every' interval of epochs,
        calculates their mean. The list is cleared and repopulated at the end of each 'update_every' interval of epochs.
    test_accuracies : List[float] 
        A list that stores testing accuracies. It accumulates accuracies over every 'update_every' interval of epochs,
        calculates their mean. The list is cleared and repopulated at the end of each 'update_every' interval of epochs.
    vis : visdom.Visdom
        Visdom visualization environment instance. 
    loss_win : visdom.Visdom.line
        Visdom window instance for the training and testing loss plots.
    eer_win : visdom.Visdom.line
        Visdom window instance for the training and tesing equal error rate plots.
    accuracy_win : visdom.Visdom.line
        Visdom window instance for the training and testing accuracy plots.
    projection_win : visdom.Visdom.matplot
        Visdom window instance for the UMAP projection of the training embeddings.
    test_projection_win : visdom.Visdom.matplot
        Visdom window instance for the UMAP projection of the testing embeddings.
    implementation_win : isdom.Visdom.text
        Visdom window instance for the implementation details.
    implementation_string : str 
        String contating the implementation details that will be written on 
        'implementation_win'.

    Methods
    -------
    log_params() -> None
        Logs the data and model's parameters to the Visdom environment.
    log_dataset(dataset: SpeakerDataset) -> None
        Logs dataset information to the Visdom environment.
    log_implementation(parameters : Dict) -> None
        Logs inmplementation details to the Visdom environment.
    update(loss: float, eer: float, accuracy: float,
            test_loss: float, test_eer: float, test_accuracy: float,
            step: int) -> None
        Updates and displays visualizations in the Visdom environment based on the provided metrics.
    draw_projections(embeddings: np.ndarray , utterances_per_speaker: int,
                    step: int, out_fpath: Union[str, Path] =None, max_speakers: int =10,
                    source_type: str ="train") -> None
        Draws two-dimensional UMAP projections of speakers embeddings and displays them
        in the Visdom environment and saves them as an .png file.
    save() -> None
        Saves the current state of the Visdom environment. 
    """
    def __init__(self, env_name:str =None, update_every:int =10,
                server:str ="http://localhost", disabled:bool =False):
        #Tracking data 
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        self.disabled = disabled
        self.accuracies = []
        self.test_losses = []
        self.test_eers = []
        self.test_accuracies = []

        print("----Updating the visualizations every %d steps " % update_every)

        if self.disabled: 
            print("----visdom is disabled, the visualizations will not be performed")
            return 

        #Set the enviroment name 
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        if env_name is None:
            self.env_name = now 
        else:
            self.env_name = "%s (%s)" % (env_name, now)
        
        #Connect to visdom and open the window in the browser 
        try:
            self.vis = visdom.Visdom(server=server, env=self.env_name, raise_exceptions=True) 
        except ConnectionError:
            raise Exception("----No visdom server detected. Run the command \"visdom\" in the command line to start it ")
        
        #Create the windows for the visdom visualizer ????
        self.loss_win = None 
        self.eer_win = None
        self.accuracy_win = None 
        self.test_projection_win = None 
        self.implementation_win = None 
        self.projection_win = None 
        self.implementation_string = ""

    def log_params(self) -> None:
        """
        Logs the data and model's parameters to the Visdom environment.
        """
        if self.disabled:
            return 
        
        param_string = "<b>Model parameters</b>:<br>" 
        for param_name in (p for p in dir(params_model) if not p.startswith("__")):
            value = getattr(params_model, param_name)
            param_string += "\t%s: %s<br> " % (param_name, value)
        
        param_string += "<b>Data parameters</b>:<br>"
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        
        self.vis.text(param_string, opts={"title": "Parameters"})

    def log_dataset(self, dataset: SpeakerDataset) -> None:
        """
        Logs dataset information to the Visdom environment. 
        The method displays the number of speakers in the dataset and outputs
        logs from the 'SpeakerDataset' class.

        Parameters
        ----------
        dataset : SpeakerDataset
            An instance of the 'SpeakerDataset' class representing a collection of speakers.
        """
        if self.disabled:
            return

        dataset_string = ""
        dataset_string += "<b>Speakers</b>: %s\n" % len(dataset.speakers)
        dataset_string += "\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={"title": "Dataset"})

    def log_implementation(self, parameters : Dict) -> None:
        """
        Logs inmplementation details to the Visdom environment.
        This method takes a dictionary of implementation parameters and their
        corresponding values and displays them in the Visdom environment. It 
        provides an overview of the training implementation setup.

        parameters : Dict
            A dictionary containing implementation details. Keys represents parameters
            names and values represent their corresponding values.

        Example
        -------
        vis.log_implementation({"Device": device_name})
        This examples logs in the Visdom environment  the device name used for training.
        """
        if self.disabled:
            return 
        
        implementation_string = ""
        for param, value in parameters.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(text=implementation_string, opts={"title": "Training implementation"})

    def update(self, loss: float, eer: float, accuracy: float,
                test_loss: float, test_eer: float, test_accuracy: float,
                step: int) -> None:
        """
        Updates and displays visualizations in the Visdom environment based on the provided metrics.
        This method tracks and accumulates the training and testing loss, equal error rate (eer) and accuracy,
        and then updates and displays the corresponding visualizations in the Visdom environment.
        It computes the mean and standard deviation of the step times, which represents the time taken
        to process each training epoch. It updates the plots for training and testing loss, EER, and accuracy
        at specified intervals defined by the 'update_every' class attribute.

        Parameters
        ----------
        loss : float
            Training loss value for the current step/epoch.
        eer : float
            Training equal error rate (eer) value for the current step/epoch.
        accuracy : float
            Training accuracy value for the current step/epoch.
        test_loss : float
            Testing loss value for the current step/epoch.
        test_eer : float
            Testing equal error rate (eer) value for the current step/epoch.
        test_accuracy : float
            Testing accuracy value for the current step/epoch.
        step: int
            Current step/epoch.

        Returns
        -------
        None
            The method updates the visualizations in the Visdom environment based on the provided metrics. 
        """
        #Update the tracking lists 
        now = timer()
        #The next line calculates the time it takes (in milliseconds) to process an epoch in the training loop
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now 
        self.losses.append(loss)
        self.eers.append(eer)
        self.accuracies.append(accuracy)
        self.test_losses.append(test_loss)
        self.test_eers.append(test_eer)
        self.test_accuracies.append(test_accuracy)

        print(".", end="")

        #Update the plots every 'update_every' steps
        if step % self.update_every != 0:
            return 
        time_string = "Step time: mean: %5dms std: %5dms " % (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\n****Step %6d Loss: %.4f, EER: %.4f, %s " % (step, np.mean(self.losses), np.mean(self.eers), time_string))
        if not self.disabled:
            self.loss_win = self.vis.line(Y=[np.mean(self.losses)], X=[step], win=self.loss_win,
                                          update="append" if self.loss_win else None, name="train loss",
                                          opts=dict(
                                                legend=["train loss", "test loss"],
                                                xlabel="Step",
                                                ylabel="Loss",
                                                title="Loss", 
                                            )
                                        )
            
            self.vis.line(Y=[np.mean(self.test_losses)], X=[step], win=self.loss_win,
                                        update="append" if self.loss_win else None, name="test loss")
            

            self.eer_win = self.vis.line(Y=[np.mean(self.eers)], X=[step], win=self.eer_win, 
                                         update="append" if self.eer_win else None, name="train eer",
                                         opts=dict(
                                                legend=["train eer", "test eer"],
                                                xlabel="Step",
                                                ylabel="Equal Error Rate - EER",
                                                title="Equal Error Rate - EER"  
                                            )
                                        )
            
            self.vis.line(Y=[np.mean(self.test_eers)], X=[step], win=self.eer_win,
                                        update="append" if self.eer_win else None, name="test eer")
            
            
            self.accuracy_win = self.vis.line(Y=[np.mean(self.accuracies)], X=[step], win=self.accuracy_win,
                                          update="append" if self.accuracy_win else None, name="train accuracy",
                                          opts=dict(
                                                legend=["train accuracy", "test accuracy"],
                                                xlabel="Step",
                                                ylabel="Accuracy",
                                                title="Accuracy", 
                                            )
                                        )
            self.vis.line(Y=[np.mean(self.test_accuracies)], X=[step], win=self.accuracy_win,
                                          update="append" if self.accuracy_win else None, name="test accuracy")
            
            if self.implementation_win is not None:
                self.vis.text(text=self.implementation_string + ("<b>%s</b> " % time_string),
                              win=self.implementation_win,
                              opts={"title": "Training implementation"}
                            )
        
        #Reset the tracking lists
        self.losses.clear() 
        self.eers.clear()
        self.step_times.clear()
        self.accuracies.clear()
        self.test_losses.clear()
        self.test_eers.clear()
        self.test_accuracies.clear()

    #def draw_projections(self, embeddings, utterances_per_speaker, step, out_fpath=None, max_speakers=speakers_per_batch):
    def draw_projections(self, embeddings: np.ndarray , utterances_per_speaker: int,
                        step: int, out_fpath: Union[str, Path] =None, max_speakers: int =10,
                        source_type: str ="train") -> None:
        """
        Draws two-dimensional UMAP projections of speakers embeddings and displays them
        in the Visdom environment and saves them as an .png file.

        The Uniform Manifold Approximation and Projection (UMAP) https://umap-learn.readthedocs.io/en/latest/ 
        is a dimensionality reduction technique that aims to capture the underlying structure and relationships
        in the data by mapping it to a lower-dimensional space.

        Parameters
        ----------
        embeddings : np.ndarray
            Numpy ndarray of floats of the current step/epoch batch embeddings.
        utterances_per_speaker : int
            Number of utterances per speaker in the batch.
        step : int
            Current training step/epoch.
        out_fpath : str, pathlib.Path, default=None
            File path to save the UMAP two-dimensional projection as an .png file.
            If not provided the projection will only be displayed in the Visdom environment. 
        max_speakers: int, default=10
            Maximum number of speakers to include in the UMAP two-dimensional projection.
        source_type : str, "train" or "test", default="train" 
            A string that takes the values of "train" or "test", speciefies
            whether the projection is from the training or testing dataset,
            it defines which of the projection windows is going to be drawn.
        """
        max_speakers = min(max_speakers, len(colormap))
        embeddings = embeddings[:max_speakers * utterances_per_speaker]

        n_speakers = len(embeddings) // utterances_per_speaker
        ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        colors = [colormap[i] for i in ground_truth]

        reducer = umap.UMAP() 
        projected = reducer.fit_transform(embeddings)   
        plt.scatter(x=projected[:, 0], y=projected[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        if not self.disabled:
            if source_type == "train":
                plt.title("UMAP Projection - step: %d " % step) 
                self.projection_win = self.vis.matplot(plt, win=self.projection_win)  
            else:
                plt.title("UMAP Test Projection - step: %d " % step)
                self.test_projection_win = self.vis.matplot(plt, win=self.test_projection_win)
        if out_fpath is not None:
            plt.savefig(out_fpath)
        plt.clf()
    
    def save(self) -> None:
        """
        Saves the current state of the Visdom environment. 
        This method triggers the saving of the current state of the Visdom environment,
        including all plots, windows, and text displays. The saved state can be reloaded 
        to restore the visualization environment to its previous state.
        """
        if not self.disabled:
            self.vis.save([self.env_name]) 