from _slash.slash_master import SlashMaster

class CommandDelegation:
    """
    Manages and pings the slash master to load commands
    """
    def __init__(self, tree, guild):
        self._slash_master = SlashMaster(tree, guild)

    def load_commands(self):
        """
        ping slash master
        """
        return self._slash_master.load_commands()