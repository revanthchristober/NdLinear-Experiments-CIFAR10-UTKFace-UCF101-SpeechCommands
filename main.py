# main.py
import argparse
import sys

# Import the runner functions from each experiment module
from ndlinear_project.cifar10.experiment import run_cifar10_experiment
from ndlinear_project.utkface.experiment import run_utkface_experiment
from ndlinear_project.ucf101.experiment import run_ucf101_experiment
from ndlinear_project.speech_commands.experiment import run_speech_commands_experiment

def main():
    parser = argparse.ArgumentParser(description="NdLinear Project Main Runner")
    parser.add_argument(
        "experiment",
        choices=["cifar10", "utkface", "ucf101", "speech_commands"],
        help="Name of the experiment to run."
    )

    # Parse only the 'experiment' argument, leave the rest for the specific experiment script
    # This prevents conflicts if sub-scripts have arguments with the same name.
    args, remaining_argv = parser.parse_known_args()

    # Reconstruct sys.argv for the sub-script's parser
    # The first element is the script name (irrelevant here), followed by the remaining args
    sys.argv = [sys.argv[0]] + remaining_argv

    print(f"--- Running Experiment: {args.experiment} ---")

    if args.experiment == "cifar10":
        run_cifar10_experiment()
    elif args.experiment == "utkface":
        # Ensure data download/unzip commands are handled, maybe add as arg/check in run_utkface_experiment
        # e.g. !curl ..., !unzip ... needs to be run beforehand or integrated.
        print("Note: UTKFace experiment assumes dataset is downloaded and unzipped.")
        run_utkface_experiment()
    elif args.experiment == "ucf101":
        print("Note: UCF101 experiment assumes dataset is downloaded and unzipped.")
        run_ucf101_experiment()
    elif args.experiment == "speech_commands":
        # The torchaudio dataset handles download automatically if needed
        run_speech_commands_experiment()
    else:
        # This case should not be reached due to 'choices' in argparse
        print(f"Error: Unknown experiment '{args.experiment}'")
        sys.exit(1)

    print(f"--- Finished Experiment: {args.experiment} ---")

if __name__ == "__main__":
    main()