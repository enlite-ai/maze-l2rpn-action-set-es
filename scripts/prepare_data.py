"""Prepares rte14 dataset with difficulty file."""
import os
import shutil

import grid2op


def prepare_rte14_data() -> None:
    """Downloads rte14 if not yet done and adds difficulty level file.
    """
    dataset_name = 'rte_case14_realistic'

    # compile full dataset path
    base_path = grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA
    dataset_path = os.path.join(base_path, dataset_name)

    if os.path.exists(dataset_path):
        print(f'Dataset {dataset_name} already downloaded.')
    else:
        # this will download the data
        grid2op.make(dataset_name)

    # check if difficulty level exists
    difficulty_file = os.path.join(dataset_path, 'difficulty_levels.json')

    if os.path.exists(difficulty_file):
        print(f'Difficulty levels file already exists.')
    else:
        shutil.copy(os.path.join('resources/', 'difficulty_levels.json'), dataset_path)
        if not os.path.exists(difficulty_file):
            print("Difficulty levels file not found in dataset directory."
                  "Make sure to run the script from the root directory.")
        else:
            print("Difficulty levels file copied successfully to dataset directory.")


if __name__ == "__main__":
    """ main """
    prepare_rte14_data()
