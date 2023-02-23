"""
When a new slash command file is made, add it here
Also add it in self._slash_commands list
"""
from _slash.add  import add  as Add
from _slash.deez import deez as Deez

class SlashMaster:
    """
    responsible for loading in each command
    """
    def __init__(self, tree, guild):
        self._tree = tree
        self._guild = guild
        # when the new command is imported add it here
        self._slash_commands = [Add, Deez,]
    
    def load_commands(self):
        """
        initialize each command with the current tree and guild
        """
        commands = []
        for command in self._slash_commands:
            pre_loaded_command = command(self._tree, self._guild)
            commands.append(pre_loaded_command)
        return commands


