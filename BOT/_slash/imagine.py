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
from _utils.embeds import generic_colored_embed
from _utils.embeds import get_imagine_embed
from _utils.stable_diffusion import (
    stable_base_json,
    Upscale,
    UpscaleModel,
    SamplerSetOne,
    SamplerSetTwo,
    call_txt2img,
    save_image,
    make_grid_image,
    process_queue,
    cooldown
)


class Imagine:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @app_commands.checks.cooldown(1, 60, key=lambda i: (i.guild_id, i.user.id))
        @tree.command(
            name="imagine",
            description="Creates images using Stable Diffusion.",
            guild=discord.Object(id=guild)
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
            sampler_set_two: SamplerSetTwo = SamplerSetTwo.none
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
            os.mkdir(f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/{stable_id}")
    
            # is stable busy? (doesn't work since the api call is blocking)
            if interaction.client._fnbb_globals.get("imagine_generating"):
                queue_item = StableQueueItem(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    quality=quality.value,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    seed=seed,
                    upscale_model=upscale_model.value,
                    sampler=sampler_set_one.value if sampler_set_one.value != None else sampler_set_two.value,
                    channel=interaction.channel_id,
                    stable_id=stable_id,
                    user=interaction.user.global_name,
                    user_avatar=interaction.user.avatar
                )
                interaction.client._fnbb_globals["imagine_queue"].add(queue_item)

                await interaction.followup.send(
                    embed=generic_colored_embed(
                        title="Adding to Queue",
                        description="An image is already in the process of generating. Adding request to the queue until stable diffusion frees up.",
                        footer_usr=interaction.user.global_name,
                        footer_img=interaction.user.avatar,
                        color="INFO",
                    )
                )
            else:

                interaction.client._fnbb_globals["imagine_generating"] = True

                # let user know we are generating image
                await interaction.followup.send(
                    embed=generic_colored_embed(
                        title="✨ Generating ✨",
                        description="Your image is being generated! It will be sent shortly. (note: higher step counts and higher quality will result in longer wait times)",
                        footer_usr=interaction.user.global_name,
                        footer_img=interaction.user.avatar,
                        footer_text="Requested by:",
                        color="SUCCESS",
                    )
                )

                # fill out the payload
                negative_prompt = negative_prompt if negative_prompt else ""
                prompt = prompt if prompt else ""

                txt2img_request_payload = stable_base_json
                txt2img_request_payload["hr_negative_prompt"] = negative_prompt
                txt2img_request_payload["negative_prompt"] = negative_prompt
                txt2img_request_payload["hr_prompt"] = prompt
                txt2img_request_payload["prompt"] = prompt
                txt2img_request_payload["hr_scale"] = quality.value
                txt2img_request_payload["cfg_scale"] = cfg_scale
                txt2img_request_payload["steps"] = steps
                txt2img_request_payload["seed"] = seed
                txt2img_request_payload["hr_upscaler"] = upscale_model.value
                if sampler_set_one.value != None:
                    txt2img_request_payload["sampler_name"] = sampler_set_one.value
                elif sampler_set_two.value != None:
                    txt2img_request_payload["sampler_name"] = sampler_set_two.value
                else:
                    await interaction.followup.send(
                        embed=generic_colored_embed(
                            title="A Sampler must be provided!",
                            description="Sampler One or Sampler Two value must be provided. If both are provided, Sampler One takes priority.",
                            footer_usr=interaction.user.global_name,
                            footer_img=interaction.user.avatar,
                            footer_text="Requested by:",
                            color="ERROR",
                        )
                    )
                    
                    interaction.client._fnbb_globals["imagine_generating"] = False
                    
                    return
                    
                try:
                    # ping local hosted stable diffusion ai
                    response = await call_txt2img(
                        payload=txt2img_request_payload
                    )

                    # save all 4 images
                    for idx, image in enumerate(response["images"]):
                        save_image(
                            path=f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/{stable_id}/{stable_id}_{idx}.png",
                            data=image
                        )
                    
                    make_grid_image(stable_id=stable_id)
                    
                    # send the grid view with buttons to upscale each picture
                    imagine_emebed, imagine_image_file = get_imagine_embed(
                        prompt=prompt,
                        negative=negative_prompt,
                        quality=quality.value,
                        cfg=cfg_scale,
                        steps=steps,
                        seed=seed,
                        footer_text="Image generated by: ",
                        footer_usr=interaction.user.global_name,
                        footer_img=interaction.user.avatar,
                        stable_id=stable_id
                    )

                    await interaction.followup.send(
                        view=ImagineView(stable_id=stable_id),
                        embed=imagine_emebed,
                        file=imagine_image_file,
                    )
                    
                    if len(interaction.client._fnbb_globals.get("imagine_queue")) == 0:
                        interaction.client._fnbb_globals["imagine_generating"] = False
                    else:
                        await process_queue(client=interaction.client)

                except ConnectionError as e:
                    interaction.client._fnbb_globals["imagine_generating"] = False
                    print(
                        DateTimeAlert(
                            text=f"STABLE DIFFUSION ERROR: {e}",
                            dtia_alert_type="ERROR",
                            message_from="bot._slash.imagine",
                        )
                    )
                    await interaction.followup.send(
                        embed=generic_colored_embed(
                            title="ERROR",
                            description="A connection error with stable-diffusion occured. Perhaps the stable-diffusion API is not running.",
                            footer_usr=interaction.user.global_name,
                            footer_img=interaction.user.avatar,
                            color="ERROR",
                        )
                    )

        @imagine.error
        async def on_imagine_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
            if isinstance(error, app_commands.CommandOnCooldown):
                cooldown_embed = generic_colored_embed(
                    title="❄️ Let's chill! ❄️",
                    description=str(error),
                    footer_text="Attempted request by:",
                    footer_usr=interaction.user.global_name,
                    footer_img=interaction.user.avatar,
                    color=0x08B6CE
                )
                await interaction.response.send_message(embed=cooldown_embed, ephemeral=False)
