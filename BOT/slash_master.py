"""
When a new slash command file is made, add it here
Also add it in self._slash_commands list
"""

import os
from datetime import datetime
from sys import platform

from _utils.alerts import ErrorAlert, GeneralAlert, InfoAlert, SuccessAlert


class SlashMaster:
    """
    responsible for loading in each command
    """

    def __init__(self, tree, guild, path, debug):
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            self._path = os.path.dirname(__file__) + "/" + path
        else:
            self._path = os.path.dirname(__file__) + "\\" + path

        self._debug = debug
        self._tree = tree
        self._guild = guild

    def load_commands(self):
        """
        initialize each command with the current tree and guild
        """
        print(f"{InfoAlert('Searching for slash commands in:')} {self._path}...")
        for file in self.get_next_command():
            print(f"loading {GeneralAlert(file).text:^10}", end=" ")
            try:
                pre_loaded_command = self.import_from(
                    f"_slash.{file}", file.capitalize()
                )
                pre_loaded_command = pre_loaded_command(self._tree, self._guild)
                print(f"{SuccessAlert('success')}")
            except Exception as error:
                print(f"{ErrorAlert('failure')}\n └─ {str(error)}")

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
