import os
import io
import json
import uuid
import typing
import base64
import asyncio
import discord
import requests
import functools

from PIL import Image
from requests.exceptions import *
from discordwebhook import Discord
from discord import app_commands
from _utils.views import ImagineView
from _utils.alerts import DateTimeAlert
from _utils.queueing import StableQueueItem
from _utils.embeds import generic_colored_embed, colors
from _utils.embeds import get_imagine_embed
from _utils.stable_diffusion import (
    Upscale,
    UpscaleModel,
    SamplerSetOne,
    SamplerSetTwo,
    Images,
)


class Imagine:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        # @app_commands.checks.cooldown(1, 60, key=lambda i: (i.guild_id, i.user.id))
        @tree.command(
            name="imagine",
            description="Creates images using Stable Diffusion.",
            guild=discord.Object(id=guild),
        )
        async def imagine(
            interaction: discord.Interaction,
            prompt: str,
            negative_prompt: str = "",
            quality: Upscale = Upscale.two,
            cfg_scale: float = 3.0,
            steps: int = 20,
            seed: int = -1,
            upscale_model: UpscaleModel = UpscaleModel.latent,
            sampler_set_one: SamplerSetOne = SamplerSetOne.ddim,
            sampler_set_two: SamplerSetTwo = SamplerSetTwo.none,
            images: Images = Images.one
        ):
            """Slash command to generate an image with Stable Diffusion.
            Stable diffusion must be running via stable-diffusion-webui with
            the `--nowebui` flag passed in to launch the webui's API

            Args:
                interaction (discord.Interaction): interaction that invoked the slash
                command
                Prompt (str): prompt to use to create the image
                Negative_Prompt (str, optional): negative prompt to use. Defaults to "".
                Quality (Upscale, optional): How much to upscale the image. Defaults to Upscale.two.
                CFG_Scale (float, optional): CFG Scale parameter. Defaults to 3.0.
                Steps (int, optional): generation steps. Defaults to 20.
                Seed (int, optional): seed to use to create the image. Defaults to -1.
            """
            await interaction.response.defer()            
            stable_id = uuid.uuid4().hex
            while stable_id in os.listdir(f"{os.getcwd()}\\BOT\\_utils\\_tmp\\stable_diffusion"):
                stable_id = uuid.uuid4().hex
            
            stable_queue_item = StableQueueItem(
                prompt=prompt,
                negative_prompt=negative_prompt,
                quality=quality.value,
                cfg_scale=cfg_scale,
                steps=steps,
                seed=seed,
                upscale_model=upscale_model.value,
                sampler=(
                    sampler_set_one.value
                    if sampler_set_one.value != None
                    else sampler_set_two.value
                ),
                channel=interaction.channel_id,
                stable_id=stable_id,
                user=interaction.user.global_name,
                user_avatar=interaction.user.avatar,
                images=images.value
            )
            
            await interaction.followup.send(embed=generic_colored_embed(
                title="Request Recieved!",
                description="You request was recieved and is being processed. Your images are on their way!",
                footer_img=interaction.user.avatar,
                footer_text="Requested by:",
                footer_usr=interaction.user.global_name,
                color=colors["SUCCESS"]
            ))
            
            await interaction.client._fnbb_globals["SCC"].delegate(stable_queue_item)
            
            
        @imagine.error
        async def on_imagine_error(
            interaction: discord.Interaction, error: app_commands.AppCommandError
        ):
            if isinstance(error, app_commands.CommandOnCooldown):
                cooldown_embed = generic_colored_embed(
                    title="❄️ Let's chill! ❄️",
                    description=str(error),
                    footer_text="Attempted request by:",
                    footer_usr=interaction.user.global_name,
                    footer_img=interaction.user.avatar,
                    color=0x08B6CE,
                )
                await interaction.response.send_message(
                    embed=cooldown_embed, ephemeral=False
                )
            else:
                print(f"UNHANDLED IMAGINE ERROR: {str(error)}")
                
                interaction.client._fnbb_globals["imagine_generating"] = False
                
                unhandled_embed = generic_colored_embed(
                    title="⚠️ Unhandled Error ⚠️",
                    description=str(error),
                    footer_text="Attempted request by:",
                    footer_usr=interaction.user.global_name,
                    footer_img=interaction.user.avatar,
                    color=colors["WARNING"],
                )
                await interaction.response.send_message(
                    embed=unhandled_embed, ephemeral=False
                )
