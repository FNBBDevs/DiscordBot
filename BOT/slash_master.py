"""
When a new slash command file is made, add it here
Also add it in self._slash_commands list
"""

import os
from datetime import datetime

class SlashMaster:
    """
    responsible for loading in each command
    """
    def __init__(self, tree, guild, path, debug):
        self._PATH = f'{os.getcwd()}/{path}'
        self._DEBUG = debug
        self._tree = tree
        self._guild = guild
    
    def load_commands(self):
        """
        initialize each command with the current tree and guild
        """
        if self._DEBUG:
            print(f'Searching for slash commands in: {self._PATH}...')        

        for file in self.get_next_command():
            if self._DEBUG:
                print(f'Attempting to load \'{file}\'...')

            try:
                pre_loaded_command = self.import_from(f'_slash.{file}', file)
                pre_loaded_command = pre_loaded_command(self._tree, self._guild)

            except Exception as e:
                # TODO: You can use an actual logging library for this! >:(
                with open(f'./error.fnbbef', 'a+') as error_file:
                    error_file.write(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} - erm... I couldn\'t load that command...  erm... the one called \'{file}\', {e}\n')
    
    def get_next_command(self):
        for file in os.listdir(self._PATH):
            if '__' not in file and "error" not in file:
                yield file[:-3]

    @staticmethod
    def import_from(module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)
        

