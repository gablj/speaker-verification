import argparse
import pathlib 

class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """
    Custom Formatter class that combines 'RawDescriptionHelpFormatter' and 'ArgumentDefaultsHelpFormatter'.
    
    'argparse.RawTextHelpFormatter' - To preserve line breaks and formatting in text
    description of arguments on the terminal (useful for multi-line descriptions).

    'argparse.ArgumentDefaultsHelpFormatter' - To include the default values of arguments.
    """
    pass 

description = " ****A driver module  for the execution of the data preprocessing. "
parser = argparse.ArgumentParser(description=description, formatter_class=CustomFormatter)
parser.add_argument("root_path", type=pathlib.Path, help=
                    "Path of the directory where the 'LibriSpeech' samples are stored,\n inside the directory where 'root_path' points the subdirectories hierarchy must be as follows: \n"
                    ".- root_path/\n"
                    "   |\n "
                    "       .- LibriSpeech/\n"
                    "           |\n"
                    "               .- train-other-500/" )
#The '--' flag is added to the name of the arguments to make it optional, meaning that
# if the user doesn't explicitly add the argument '--myarg' on the command line 
# then 'myarg' will take the default value defined in 'add_argument("--myarg", default=a_default_value)'
parser.add_argument("-o", "--out_path", type=pathlib.Path, default=argparse.SUPPRESS,
                    help="Path of the directory where the .npy files containing the mel-frequency energy vectors \n"
                    "of the speakers will be saved.")

parser.add_argument("--skip_existing", type=bool, default=True,
                    help="If 'False' will re-compute already preprocessed audiofiles, i.e., \n"
                        "will delete existing .npy files and recompute mel-frequency energy vectors \n"
                         "for the complete dataset, If 'True' will skip existing ones.")
                           
args = parser.parse_args()
print(args)
