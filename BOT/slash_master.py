"""
When a new slash command file is made, add it here
Also add it in self._slash_commands list
"""

import os

from _utils.alerts import GeneralAlert, SuccessAlert, InfoAlert, ErrorAlert


class SlashMaster:
    """
    responsible for loading in each command
    """

    def __init__(self, tree, guild, path, debug):
        self._path = f"{os.path.dirname(__file__)}\{path}"
        self._debug = debug
        self._tree = tree
        self._guild = guild

    def load_commands(self):
        """
        initialize each command with the current tree and guild
        """
        if self._debug:
            print(f"{InfoAlert('Searching for slash commands in:')} {self._path}...")
            print(f"{GeneralAlert('Loading commands . . .')}")
        for file in self.get_next_command():
            print(f"loading {GeneralAlert(file)} . . .")
            try:
                pre_loaded_command = self.import_from(
                    f"_slash.{file}", file.capitalize()
                )
                pre_loaded_command = pre_loaded_command(self._tree, self._guild)
                print(f" ┗ {SuccessAlert('load was successful')}")
            except ImportError as error:
                print(f" ┗ {ErrorAlert('error encountered')}: {str(error)}")

    def get_next_command(self):
        """
        Description: TODO
        """
        for file in os.listdir(self._path):
            if "__" not in file and "error" not in file:
                yield file[:-3]

    @staticmethod
    def import_from(module, name):
        """
        Description: TODO
        """
        module = __import__(module, fromlist=[name])
        return getattr(module, name)
