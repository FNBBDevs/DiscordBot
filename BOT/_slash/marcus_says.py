import enum
import os

import discord
from discordwebhook import Discord


class MarcusSays:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree: discord.app_commands.CommandTree, guild: str, args=None):
        """
        Description: Constructor.
        """

        # args is passed in from slash_master. slash_master recieved args
        # args from the fortniteballsbot which was a list of the text_channels
        # in the guild. args[0] is a list of text_channel objects in the guild
        self.ChannelEnum = enum.Enum(
            "ChannelEnum", {channel.name: channel for channel in args[0]}
        )

        self.marcus = Discord(url=os.getenv("MARCUS"))
        self.marcus_id = int(os.getenv("MARCUS_ID"))

        @tree.command(
            name="marcus_says",
            description="Have marcus say something in a channel of your choice.",
            guild=discord.Object(id=guild),
        )
        async def MarcusSays(
            interaction: discord.Interaction, channel: self.ChannelEnum, message: str
        ):
            """
            The MarcusSays command. Allows the user to choose a channel and a message for Marcus to say in that channel.
            """

            marcus_old_channel = None
            marcus_webhook = None

            await interaction.response.defer(ephemeral=True)

            # find the marcus webhook
            if self.marcus_id:
                webhooks = await interaction.guild.webhooks()
                for webhook in webhooks:
                    if webhook.id == self.marcus_id:
                        marcus_old_channel = webhook.channel
                        marcus_webhook = webhook

            # point marcus to the user selected channel
            await marcus_webhook.edit(channel=channel.value)

            # have marcus say the provided message
            self.marcus.post(content=message)

            # set marcus to look back at the his original channel
            await marcus_webhook.edit(channel=marcus_old_channel)

            # delete that message
            original_response = await interaction.original_response()
            await original_response.delete()
