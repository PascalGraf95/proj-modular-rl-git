from modules.misc.model_path_handling import remove_old_checkpoints
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model_path', help="Enter the model dictionary in which old checkpoints shall be "
                                                    "removed.",
                        type=str, required=True, default=".")
    args = parser.parse_args()
    remove_old_checkpoints(args.model_path)


if __name__ == '__main__':
    main()