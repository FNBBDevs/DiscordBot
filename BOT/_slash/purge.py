import discord


class Purge:
    def __init__(self, tree, guild, args=None):
        @tree.command(
            description="clear specified number of messages from the chat",
            guild=discord.Object(id=guild),
        )
        async def purge(
            interaction: discord.Interaction,
            count: discord.app_commands.Range[int, 1, 100],
        ):
            await interaction.response.defer(ephemeral=True, thinking=True)
            await interaction.channel.purge(limit=count)

            # Delete follow-up message after 3 seconds
            msg: discord.WebhookMessage = await interaction.followup.send(
                f"Deleted {count} messages from the chat...", ephemeral=True
            )
            await msg.delete(delay=3)

        @tree.command(
            name="purge-channel",
            description="delete all messages from the channel",
            guild=discord.Object(id=guild),
        )
        async def purge_channel(interaction: discord.Interaction):
            webhooks = await interaction.channel.webhooks()
            await interaction.response.send_message(
                content="This channel will be deleted...", ephemeral=True
            )
            new_channel = await interaction.channel.clone(reason="Has been nuked")
            for webhook in webhooks:
                await webhook.edit(channel=new_channel)
            await interaction.channel.delete()
