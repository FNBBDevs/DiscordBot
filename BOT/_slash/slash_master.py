"""
When a new slash command file is made, add it here
Also add it in self._slash_commands list
"""

import os
from datetime import datetime
from Config import SLASH_PATH
from importlib import import_module

class SlashMaster:
    """
    responsible for loading in each command
    """
    def __init__(self, tree, guild):
        self._tree = tree
        self._guild = guild
        self._slash_commands = self.get_commands()
    
    def load_commands(self):
        """
        initialize each command with the current tree and guild
        """
        commands = []
        for command in self._slash_commands:
            try:
                pre_loaded_command = self.import_from(f'_slash.{command}', command)
                pre_loaded_command = pre_loaded_command(self._tree, self._guild)
                commands.append(pre_loaded_command)
            except:
                with open(f'{SLASH_PATH}/error.fnbbef', 'a+') as error_file:
                    error_file.write(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} - erm... I couldn\'t load that command...  erm... the one called \'{command}\', maybe try again . . .\n')
        return commands
    
    def import_from(self, module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)
    
    def get_commands(self):
        valid_files = [val[:-3] for val in os.listdir(SLASH_PATH) if '__' not in val and val != 'slash_master.py' and "error" not in val]
        return valid_files
        

