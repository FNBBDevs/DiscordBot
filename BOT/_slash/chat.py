import os
import json
import time
import discord
import requests
from discord.ext import commands
from _utils.openaiprompter import OpenAIPrompter
from discordwebhook import Discord
from profanity_check import predict, predict_prob
from discord import app_commands
from _utils.alerts import DateTimeAlert


class Chat:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @tree.command(
            name="chat",
            description="chat with Marcus!",
            guild=discord.Object(id=guild),
        )
        async def chat(interaction: discord.Interaction, message: str):
            """
            /add command
            :param a: message for Marcus
            :param b: password to utilize the service
            """

            await interaction.response.defer()

            print(
                DateTimeAlert(
                    f"{interaction.user.global_name} sent the following to the chat command: '{message}'",
                    dtia_alert_type="INFO",
                    message_from="bot._slash.chat",
                ).text,
            )

            users_name = json.loads(os.getenv("ID_TO_NAME")).get(str(interaction.user.id))
            if not users_name: users_name = interaction.user.global_name

            marcus = Discord(url=os.getenv("MARCUS"))
            marcus_id = int(os.getenv("MARCUS_ID"))
            
            # if interaction.user.global_name not in interaction.client._fnbb_globals.get("mh"):
            #     interaction.client._fnbb_globals["mh"][interaction.user.global_name] = []
            
            # if len(interaction.client._fnbb_globals["mh"][interaction.user.global_name]) > 0:
            #     prompt += "Chat history:\n"
            #     for um, bm in interaction.client._fnbb_globals["mh"][interaction.user.global_name]:
            #         prompt += f"{interaction.user.global_name}: {um}\nYou (Marcus): {bm}\n\n"

            #     prompt += f"Respond to {interaction.user.global_name}'s message: "
                
            if marcus_id:
                webhooks = await interaction.guild.webhooks()

            # update marcus to response where the command was called
            if marcus_id:
                for webhook in webhooks:
                    if webhook.id == marcus_id:
                        await webhook.edit(channel=interaction.channel)

            try:
                # marcus_should_say = OpenAIPrompter().complete(
                #     prompt=message
                # )
                response = requests.post(
                    "http://127.0.0.1:8000/chat/ollama",
                    data=json.dumps({"model": "marcus", "prompt": f"Hey it's {users_name}, {message}"}),
                ).json()
                marcus_should_say = response["data"]["response"]
                
                # interaction.client._fnbb_globals["mh"][interaction.user.global_name].append((message, marcus_should_say))
                # if len(interaction.client._fnbb_globals["mh"][interaction.user.global_name]) > 5:
                #     interaction.client._fnbb_globals["mh"][interaction.user.global_name] = interaction.client._fnbb_globals["mh"][interaction.user.global_name][1:]

                if marcus_should_say is not None:
                    await interaction.followup.send(content=message)
                    # marcus.post(content=marcus_should_say.content)
                    for chunk in range(0, len(marcus_should_say), 2000):
                        marcus.post(content=marcus_should_say[chunk:chunk + 2000])
                else:
                    await interaction.followup.send(content="ERROR")
            except Exception as exception:
                print(f"[ERROR] /chat: {exception}")
                await interaction.followup.send(content="ERROR")

        @chat.error
        async def on_chat_error(
            interaction: discord.Interaction, error: app_commands.AppCommandError
        ):
            if isinstance(error, app_commands.CommandOnCooldown):
                await interaction.response.send_message(content=str(error))
            else:
                print(DateTimeAlert(text=str(error), dtia_alert_type="ERROR", message_from="BOT._slash.chat"))
