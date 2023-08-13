"""
This script serves as a driver module for the execution of the preprocessing of speech samples
from LibriSpeech's 'train-other-500' dataset.

Usage: (the brackets indicate that the argument is optional when using the command-line) 
-----
'python preprocess_driver.py root_path [--out_path out_directory] [--skip_existing] [--no_trim_silence]'

For a broader description of the arguments and usage:
'python preprocess_driver.py -h'

Arguments:
---------
root_path : str 
    The path where the LibriSpeech 'train-other-500' directory is located.
    The subdirectory structure within this directory should remain unchanged from
    its layout when initially unzipped.
--out_path : str, Optional 
    Path of the directory where the .npy files containing the mel-frequency energy vectors
    of the speakers' speech samples will be saved.
    If '-out_path' is not passed, a directory named 'data_out' will be created inside 'root_path'.
--skip_existing : Optional, deafult=False
    If 'True', re-computes preprocessed audiofiles. Deletes existing .npy files and re-computes
    mel-frequency energy vectors for the complete dataset.     
--no_trim_silence : Optional, default=False
    If 'True', audio files will not be trimmed for long silences. 
"""

import argparse
import pathlib 
from multiprocessing import freeze_support 

from sencoder.preprocess_data import PreprocessData

if __name__ == '__main__':
    freeze_support()
    class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        """
        Custom Formatter class that combines 'RawDescriptionHelpFormatter' and 'ArgumentDefaultsHelpFormatter'.
        
        'argparse.RawTextHelpFormatter' - To preserve line breaks and formatting in text
        description of arguments on the terminal (useful for multi-line descriptions).

        'argparse.ArgumentDefaultsHelpFormatter' - To include the default values of arguments.
        """
        pass 

    description = """ ******A driver module for the data preprocessing of the speech samples from LibriSpeech's 'train-other-500' dataset,
    the arguments that must be passed to the command line, along with 'python preprocess_driver.py', are:
    'root_path' (not optional) , '--out_path' (optional), '--skip_existing' (optional), '--no_trim_silence' (optional).
    ****For a complete description of the command line arguments use:
    'python preprocess_driver.py -h'  """
    parser = argparse.ArgumentParser(description=description, formatter_class=CustomFormatter)
    # The '--' flag is added to the name of the arguments to make it optional, meaning that
    # if the user doesn't explicitly add the argument '--myarg' on the command line 
    # then 'myarg' will take the default value defined in 'add_argument("--myarg", default=a_default_value)'
    parser.add_argument("root_path", type=pathlib.Path, help=
                        "The path where the LibriSpeech 'train-other-500' directory is located.\n"
                        " The subdirectory structure within this directory should remain unchanged from \n"
                        " its layout when initially unzipped. \n"
                        " It follows a specific pattern, where each speaker is assigned a subdirectory \n"
                        " which in turn contains subdirectories for their speech samples: \n"
                        ".- train-other-500/\n"
                        "   |\n"
                        "   .-- speaker_idx/\n"
                        "   |        |\n"
                        "   |        .-- speaker_idx_samples1/\n"
                        "   |        .-- speaker_idx_samples2/\n"
                        "   |        .--         ...            \n"
                        "   .-- speaker_idy/\n"
                        "   |         |\n"
                        "   |        .-- speaker_idy_samples1/\n"
                        "   |        .--          ...           \n"
                        "   .--... \n"
                        "\nNOTE: 'root_path' argument is not optional."  )
    parser.add_argument("--out_path", type=pathlib.Path, default=argparse.SUPPRESS,
                        help="Path of the directory where the .npy files containing the mel-frequency energy vectors \n"
                        "of the speakers' speech samples will be saved. \n"
                        "If '-out_path' is not passed, then a directory named 'data_out' will be created inside 'root_path'.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="If 'False', will re-compute already preprocessed audiofiles, i.e., \n"
                            "will delete existing .npy files and recompute mel-frequency energy vectors \n"
                            "for the complete dataset#. If 'True', will skip existing ones. \n"
                            "NOTE: Default is 'False', to change it to 'True' add the flag '--skip_existing' \n"
                            "on the command line without any added values: \n"
                            "'python preprocess_driver.py root_path -out_path --skip-existing'")
    parser.add_argument("--no_trim_silence", action="store_true",
                        help="If 'False', trim long silences from the audio files. \n"
                            "NOTE: Default is 'False', to change it to 'True' add the flag '--no_trim_silence' \n"
                            "on the command line without any added values: \n"
                            "'python preprocess_driver.py root_path -out_path --no_trim_silence'")

    args = parser.parse_args()
    print(args)

    # If no output directory was given, creates one named 'data_out' in 'root_path'
    if not hasattr(args, "out_path"):
        args.out_path = args.root_path.joinpath("data_out")
        print("No '--out_path' given, the preprocessed .npy files will be saved at 'root_path/data_out'")
    args.out_path.mkdir(exist_ok=True)

    # Convert the namespace object created by 'argparse' to a dictionary
    # with key-values in the form 'argument:value_passed_on_command_line'
    args = vars(args)

    preprocess = PreprocessData(args['root_path'], args['out_path'])
    preprocess.preprocess_data(args['skip_existing'])
