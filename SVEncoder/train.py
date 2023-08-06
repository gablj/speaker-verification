import numpy as np
import torch 
from sklearn.metrics import accuracy_score
from pathlib import Path
import json
import os

import params_model
from speaker_dataset_dataloader import SpeakerDataset, SpeakerDataLoader 
from model import Encoder
from visualization import Visualizations
from profiler import Profiler

'''
"GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION" - https://arxiv.org/pdf/1710.10467.pdf [1]
'''

def sync(device: torch.device) -> None:
    """
    This functions ensures that all CUDA operations on the given device are complete before continuing
    with the next instructions. This is useful when dealing with multi-GPU systems to ensure data 
    consistency and synchronization across different GPU devices.

    Parameters
    ----------
    device : torch.device
        The device in which CUDA operations need to be synchronized.
        If the device is a "cpu" this function does nothing, as CPU
        operations are already synchronous.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(run_id: str, loss_method: str, preprocessed_data_train_path: Path,
        preprocessed_data_test_path: Path,  models_dir: Path,
        num_workers: int, umap_every: int, save_every: int, 
        backup_every: int, vis_every: int, force_restart: bool,
        no_visdom: bool, visdom_server: str ="http://localhost") -> None:
    """
    Trains the speaker verification model.
    
    Parameters
    ----------
    run_id : str 
        A unique identifier for the current training run.
    loss_method : str
        The method used for loss computation, either "softmax" as per eq.(6) from [1] or
        "contrast" as per eq.(7) from [1].
    preprocessed_data_train_path : Path
        A 'pathlib.Path' instance of the directory where the train split of all the speakers'
        subdirectories are located, each subdirectory contains the .npy files of the preprocessed
        mel frequency energies of the frames corresponding to the speaker's utterances.
    preprocessed_data_test_path : Path
        A 'pathlib.Path' instance of the directory where the test split of all the speakers'
        subdirectories are located, each subdirectory contains the .npy files of the preprocessed
        mel frequency energies of the frames corresponding to the speaker's utterances.
    models_dir : Path
        A 'pathlib.Path' instance of the directory where the trained models will be saved.
    num_workers : int
        The number of worker threads to use for data loading.
    umap_every : int
        The frequency (in steps) at which to draw the UMAP 2-dimensional embeddings projections and save them,
        set to 0 to disable UMAP visulization.
    save_every : int
        The frequency (in steps) at which to save the model checkpoint, set to 0 to disable saving the model 
        during training.
    vis_every : int
        The frequency (in steps) at which to update the visualizations on visdom, this is also the frequency
        in which the loss, accuracy and eer are going to be averaged.
    force_restart : bool
        If True, deletes any existing model with id 'run_id' on directory 'models_dir' and trains from scratch,
        If False, the training will resume from the existing model checkpoint, if available.
    no_visdom : bool
        If True, disables visdom visualization.
    visdom_server : str, default = "http://localhost"
        The address of the visdom server for visualization.
    """
    #Initiate train dataset and dataloader 
    dataset = SpeakerDataset(preprocessed_data_train_path)
    loader = SpeakerDataLoader(dataset=dataset,
                          speakers_per_batch=params_model.speakers_per_batch,
                          utterances_per_speaker=params_model.utterances_per_speaker,
                          num_workers=num_workers - num_workers//2)
    
    #Inititate test dataset and dataloader
    dataset_test = SpeakerDataset(preprocessed_data_test_path)
    loader_test = SpeakerDataLoader(dataset=dataset_test, 
                               speakers_per_batch=params_model.speakers_per_test_batch,
                               utterances_per_speaker=params_model.utterances_per_test_speaker,
                               num_workers=num_workers//2)
    iterloader_test = iter(loader_test) 
    
    
    #Setup the device on which to run the forward pass and the loss, these can differ, 
    #because the forward pass is faster on the GPU but the loss is often, depending on the 
    #hyperparameters, faster on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Create the model and the optimizer 
    model = Encoder(device, loss_device, loss_method)
    optimizer = torch.optim.Adam(model.parameters(), lr=params_model.learning_rate_init) 
    init_step = 1

    #Configure file path for the model 
    model_dir = models_dir / run_id #path where the trained models will be saved 
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt" 

    #Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("----Found existing model \"%s\", loading it and resuming training " % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = params_model.learning_rate_init
        else:
            print("----No model \"%s\" was found in the pointing directory, starting training from zero " % run_id)
        
        if not os.path.isfile(model_dir / f"{run_id}_data.json"): #Check wether a backup file exists, if not, it creaates it 
            with open(model_dir / f"{run_id}_data.json", "w") as f:
                f.write('{}')
        
        if not os.path.isfile(model_dir / f"{run_id}_test_data.json"): #Check wether a backup file exists, if not, it creaates it 
            with open(model_dir / f"{run_id}_test_data.json", "w") as f:
                f.write('{}')
    else:
        print("----'force_restart' set to 'True'. Any existing model in the passed directory will be overwritten.")
        print("----Starting training from zero")

        if os.path.isfile(model_dir / f"{run_id}_data.json"): #Check wether a backup file exists, if yes, rewrites it  
            with open(model_dir / f"{run_id}_data.json", "w") as f:
                f.write('{}')
        
        if os.path.isfile(model_dir / f"{run_id}_test_data.json"): #Check wether a backup file exists, if yes, rewrites it  
            with open(model_dir / f"{run_id}_test_data.json", "w") as f:
                f.write('{}')
    
    #Trained the model from zero, or resume training, depending on the value of 'force-restart'
    #model.train() #Check documentation, torch method to put the model in training mode

    #Initialize the visdom visualization enviroment
    vis = Visualizations(env_name=run_id, update_every=vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})

    print("----The training process will stop until manually stopped \ctrl + C\ ")
    print("there are %d workers " % num_workers)
    print("using %s loss method " % loss_method)
    #The next two dictionaries are created to create a backup .json file of how training and testing 
    #losses and metrics evolve along the training, for plotting and drawing conclusions.
    data_backup = {}
    data_test_backup = {}

    profiler = Profiler(summarize_every=10, disabled=False)
    #Training loop
    for step, (speaker_batch, speaker_test_batch) in enumerate(zip(loader, loader_test), init_step): 
        model.train() #Added inside loop, since is "turned off" in each iteration, when evaluating
        profiler.tick("Blocking, waiting for batch to be loaded in a separate thread")

        #Forward pass 
        inputs = torch.from_numpy(speaker_batch.data).to(device) #gets the output of the batch of utterances as a tensor, has shape '(speaker_per_batch * utterances_per_speaker, n_frames, n_mels)'
        sync(device)        
        profiler.tick("Data to %s " % device)
        embeddings = model.forward(inputs)
        sync(device)        
        profiler.tick("Forward pass")
        embeddings_loss = embeddings.view((params_model.speakers_per_batch, params_model.utterances_per_speaker, -1)).to(loss_device) #reshaping is necessary for computing similarity matrix as defined in "model.py"
        predictions = model.similarity_matrix(embeddings_loss) #****TODO, similarity matrix gets computed twice, one here and the other time in the loss function
        loss, eer = model.loss(embeddings_loss)
        sync(loss_device)   
        profiler.tick("Loss")

        #Backward pass
        model.zero_grad()
        loss.backward()
        profiler.tick("Backward pass") 
        model.do_gradient_ops() 
        optimizer.step()
        profiler.tick("Parameter update")

        #Update train dictionary with train loss and metrics
        model.eval()
        targets = np.repeat(np.arange(params_model.speakers_per_batch), params_model.utterances_per_speaker)
        predictions = predictions.reshape((params_model.speakers_per_batch * params_model.utterances_per_speaker), params_model.speakers_per_batch)
        accuracy = accuracy_score(np.argmax(predictions.detach().cpu().numpy(), axis=1 ), targets)
        data_backup[step] = {
            "loss": loss.item(),
            "eer": eer,
            "accuracy": accuracy,
        }

        #Evaluate on the test dataset 
        with torch.no_grad():
            speaker_test_batch = torch.from_numpy(speaker_test_batch.data).to(device)
            sync(device)
            test_targets = np.repeat(np.arange(params_model.speakers_per_test_batch), params_model.utterances_per_test_speaker)
            test_embeddings = model.forward(speaker_test_batch)
            sync(device)
            test_embeddings_loss = test_embeddings.view((params_model.speakers_per_test_batch, params_model.utterances_per_test_speaker, -1)).to(loss_device)
            test_predictions = model.similarity_matrix(test_embeddings_loss)
            sync(loss_device)
            test_predictions = test_predictions.reshape((params_model.speakers_per_test_batch * params_model.utterances_per_test_speaker), params_model.speakers_per_test_batch)
            test_loss, test_eer = model.loss(test_embeddings_loss)
            sync(loss_device)
            test_accuracy = accuracy_score(np.argmax(test_predictions.detach().cpu().numpy(), axis=1 ), test_targets)
            
            #Update test dictionary with test loss and metrics
            data_test_backup[step] = {
                "loss_test": test_loss.item(),
                "eer_test": test_eer,
                "test_accuracy": test_accuracy,
            }

            #Update .json backup files of train and testing loss and metrics
            if step % vis_every == 0:
                avg_loss = sum(val['loss'] for val in data_backup.values()) / vis_every
                avg_eer = sum(val['eer'] for val in data_backup.values()) / vis_every
                avg_accuracy = sum(val['accuracy'] for val in data_backup.values()) / vis_every
                averages = {step:{'loss': avg_loss, 'eer': avg_eer, 'accuracy':avg_accuracy}}
                with open(model_dir / f"{run_id}_data.json", "r") as f:
                    existing_data = json.load(f)
                existing_data.update(averages)
                with open(model_dir / f"{run_id}_data.json", "w") as f:
                    json.dump(existing_data, f)
                data_backup.clear()

                avg_test_loss = sum(val['loss_test'] for val in data_test_backup.values()) / vis_every
                avg_test_eer = sum(val['eer_test'] for val in data_test_backup.values()) / vis_every
                avg_test_accuracy = sum(val['test_accuracy'] for val in data_test_backup.values()) / vis_every
                averages = {step:{'loss_test': avg_test_loss, 'eer_test': avg_test_eer, 'test_accuracy':avg_test_accuracy}}
                with open(model_dir / f"{run_id}_test_data.json", "r") as f:
                    existing_data = json.load(f)
                existing_data.update(averages)
                with open(model_dir / f"{run_id}_test_data.json", "w") as f:
                    json.dump(existing_data, f)
                data_test_backup.clear()

        #Update visualizations on visdom
        vis.update(loss.item(), eer, accuracy, test_loss.item(), test_eer, test_accuracy, step)

        #Draw UMAP 2dimensional embedding projections and save them to the backup folder 
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing projections and saving them, step %d " % step)
            projection_fpath = model_dir / f"umap_{step:06d}.png"           
            embeddings = embeddings.detach().cpu().numpy() #transforming the "embeddings" tensor to a numpy array
            vis.draw_projections(embeddings, params_model.utterances_per_speaker, step, projection_fpath, source_type="train")

            test_projection_fpath = model_dir / f"test_umap_{step:06d}.png"           
            test_embeddings = test_embeddings.detach().cpu().numpy() #transforming the "embeddings" tensor to a numpy array
            vis.draw_projections(test_embeddings, params_model.utterances_per_test_speaker, step, test_projection_fpath, source_type="test")
        
            vis.save()
        
        #Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0: 
            print("Saving the model, step %d " % step)
            torch.save(obj={"step": step + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                       f=state_fpath)
            
        #Make a backup of the model and train and test losses and metrics 
        if backup_every != 0 and step % backup_every == 0: 
            print("Making a backup, step %d " % step)  
            backup_fpath = model_dir / f"encoder_{step:06d}.bak"
            torch.save(obj={"step": step + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                       f=backup_fpath)
            
            #since "backup_every" is a multiple of "vis_every", then file "{run_id}_data.json" already exist
            with open(model_dir / f"{run_id}_data.json", "r") as f:
                existing_data = json.load(f)
            with open(model_dir / f"{run_id}_data_backup_{step:06d}.json", "w") as f:
                json.dump(existing_data, f)

            with open(model_dir / f"{run_id}_test_data.json", "r") as f:
                existing_data = json.load(f)
            with open(model_dir / f"{run_id}_test_data_backup_{step:06d}.json", "w") as f:
                json.dump(existing_data, f)

        profiler.tick("Visualizations, saving model and saving backup.")
