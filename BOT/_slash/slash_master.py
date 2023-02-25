"""
When a new slash command file is made, add it here
Also add it in self._slash_commands list
"""

import os
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
            pre_loaded_command = self.import_from(f'_slash.{command}', command)
            pre_loaded_command = pre_loaded_command(self._tree, self._guild)
            commands.append(pre_loaded_command)
        return commands
    
    def import_from(self, module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)
    
    def get_commands(self):
        valid_files = [val[:-3] for val in os.listdir(SLASH_PATH) if '__' not in val and val != 'slash_master.py']
        return valid_files
        

