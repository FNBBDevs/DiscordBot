"""
When a new slash command file is made, add it here
Also add it in self._slash_commands list
"""

import os
from sys import platform

from _utils.alerts import ErrorAlert, GeneralAlert, InfoAlert, SuccessAlert, DateTimeAlert


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

    def load_commands(self, args=None):
        """
        initialize each command with the current tree and guild
        """
        for file in self.get_next_command():
            print(DateTimeAlert(f"loading {file} . . . ",
                                dtia_alert_type="INFO",
                                message_from="bot.slash_master").text,
                                end="")
            try:
                file_capitalized = "".join([word.capitalize() for word in file.split("_")])
                pre_loaded_command = self.import_from(
                    f"_slash.{file}", file_capitalized
                )
                pre_loaded_command = pre_loaded_command(self._tree, self._guild, args=(args[0],))
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
