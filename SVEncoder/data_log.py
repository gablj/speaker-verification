from pathlib import Path 
from typing import List
from datetime import datetime

import numpy as np 

import params_data

class DataLog:
    """
    A class used to create a log as .txt file about the metadata from the dataset.
    
    Attributes
    ----------
    out_path : Path 
        A pathlib.Path instance of the output directory, where the preprocessed output files
        will get stored.
    data_name : str 
        The name of the directory/dataset where the audio files are stored, i.e., not the complete path
        but the directory/folder name. 
    text_file : file object of type 'TextIOWrapper' for writing 
        It represents the opened txt file where the log is written.
    sample_data : dict
        To store statistics from the preprocessing execution.

    Methods
    -------
    write(line: str)
        Writes the string 'line' into the log txt file. 
    add_sample(**kwargs)
        Adds key-argument as statistics sample to 'sample_data' dict. 
    finalize()
        Writes the concluding statistics of the process to the log file and closes the log file object.
    """
    def __init__(self, out_path: Path, data_name: str ):
        """
        Parameters
        ----------
        out_path: Path 
            A pathlib.Path instance of the output directory, where the preprocessed output files
            will get stored. 
        data_name : str 
            The name of the directory/dataset where the audio files are stored, i.e., not the
            complete path but the directory/folder name.
        """
        self.text_file = open(Path(out_path, "log_%s.txt" % data_name.replace("/", "_")), "w") #Creates txt file and opens it for writing 
        self.sample_data = dict()
        
        start_t = datetime.now().strftime("%d/%m/%y, %H:%M")
        self.write("Creating dataset %s, starting at %s" % (data_name, start_t))
        self.write("----")
        self.__log_params()
        
    def __log_params(self) -> None:
        """
        Private method that writes the data parameters from 'params_data.py' into
        the txt log file. 
        """
        self.write("Parameters: ")
        for parameter in (s for s in dir(params_data) if not s.startswith("__")):
            val = getattr(params_data, parameter) #since "type(parameter) = <class 'module'>", can use "getattr(object, attribute)" - returns the value of the attribute from the object
            self.write("%s = %d" % (parameter, val))
        self.write("----")
        
    def write(self, line: str) -> None:
        """
        Writes a string to the log file object attribute 'text_file'.

        Parameters
        ----------
        line : str 
            The string to be written to the txt log file.
        """
        self.text_file.write("%s\n" % line)

    def add_sample(self, **kwargs) -> None:
        """
        Adds key-argument as statistics sample to 'sample_data' dict. 
        
        Parameters
        ----------
        **kwargs : key-value pairs
            Keyword arguments representing statistics samples to be added 
            to the data log .txt file, along with their corresponding
            sample values.
        """
        for param, value in kwargs.items():
            if not param in self.sample_data: 
                self.sample_data[param] = []
            self.sample_data[param].append(value)
        
    def finalize(self) -> None:
        """
        Writes the concluding statistics of the process to the log .txt file
        and closes the log file object.
        """
        self.write("----")
        self.write("Statistics: ")
        for param_name, values in self.sample_data.items():
            self.write("\t%s: " % param_name)
            self.write("\t\tmin %.3f, max %.3f " % (np.min(values), np.max(values)))
            self.write("\t\tmean %.3f, median %.3f " % (np.mean(values), np.median(values)))
        self.write("----")
        end_t = datetime.now().strftime("%d/%m/%y, %H:%M")
        self.write("Finished at %s" % end_t)
        self.text_file.close()
