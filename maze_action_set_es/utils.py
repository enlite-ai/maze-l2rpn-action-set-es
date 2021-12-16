"""Contains some utilities."""
import os


class SwitchWorkingDirectory:
    """Switches to the provided target directory and back to the original directory on exit.
    Convenient for instantiating trained models from an output directory.

    :param target_dir: The target directory to switch to.
    """

    def __init__(self, target_dir: str):
        self.input_dir = target_dir

    def __enter__(self):
        self.original_dir = os.getcwd()
        print(f'Switching load directory to {self.input_dir}')
        os.chdir(self.input_dir)

    def __exit__(self, *args):
        os.chdir(self.original_dir)
