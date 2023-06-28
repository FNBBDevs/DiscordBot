import re


class Contains:
    """
    This class simply serves to check whether a message contains a series of
    defined flags and returns the appropriate response
    """

    def __init__(self):
        self.flags = {
            "fortniteballs": "Fortnite balls\nhttps://www.youtube.com/watch?v=Kodx9em0mXE&ab_channel=Sergeantstinky-Topic",
            "lol": "https://i.imgflip.com/7b8363.gif",
            "bruhshell": "! ! ! BRUH SHELL IS A CRYPTO-MINING SPYWARE ! ! !",
            "etchris": "https://soundcloud.com/etchris?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing",
            "c++": "C++ Dev be like:\nhttps://tenor.com/view/giga-moai-gif-26488016",
            "python": "Python Dev be like :P\nPwease use python i can't understand anything besides python\nhttps://tenor.com/view/gun-tears-cat-point-gun-crying-cat-gif-17741904",
        }

    def execute(self, message_in, debug):
        """
        Given a message, this function looks for certain phrases and
        returns the proper response given it exists in the message
        :param message_in: incoming message from discord
        :param debug     : to print debug messages
        :param out       : the resulting responses the bot should return
        """
        filter = re.compile("[^a-zA-Z+]")
        message_in = filter.sub("", message_in)
        out = []

        if debug:
            print(
                f"[Contains]: incoming message -> '{message_in}'\n[Contains]: checking flags . . ."
            )
        for flag in self.flags:
            if debug:
                print(f"\t{flag:<20s}: ", end="")
            if flag in message_in:
                if debug:
                    print(True)
                out.append(self.flags[flag])
            else:
                if debug:
                    print(False)
        return out
