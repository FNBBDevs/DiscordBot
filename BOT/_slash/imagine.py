import os
import discord
from discordwebhook import Discord
from _utils.views import ImagineView
from _utils.embeds import generic_colored_embed
from _utils.stable_diffusion import stable_base_json
import io
import json
import base64
import requests
from requests.exceptions import *
from enum import Enum

class Upscale(Enum):
    one = 1
    two = 2
    three = 3
    four = 4

class Imagine:
    """
    Description: adds two numbers.
    """

    def __init__(self, tree, guild, args=None):
        """
        Description: Constructor.
        """

        @tree.command(
            name="imagine",
            description="Creates images with Midjourney",
            guild=discord.Object(id=guild),
        )
        async def imagine(interaction: discord.Interaction, prompt: str, negative_prompt: str = "", quality: Upscale = Upscale.two):
            """
            /imagine
            """
            
            await interaction.response.defer()

            if interaction.client._fnbb_globals.get("imagine_generating"):
                
                await interaction.followup.send(embed=generic_colored_embed(
                    title="Adding to Queue",
                    description="An image is already in the process of generating. Add prompts to the queue until stable diffusion frees up.",
                    footer_usr=interaction.user.global_name,
                    footer_img=interaction.user.avatar,
                    color="SUCCESS"
                ))
                
            else:
                
                interaction.client._fnbb_globals["imagine_generating"] = True
                
                await interaction.followup.send(embed=generic_colored_embed(
                        title="Generating!",
                        description="Your image is being generated! It will be sent shortly. (Estimated wait: 1min)",
                        footer_usr=interaction.user.global_name,
                        footer_img=interaction.user.avatar,
                        color="SUCCESS"
                ))
                
                negative_prompt = negative_prompt if negative_prompt else ""
                prompt = prompt if prompt else ""
                
                txt2img_request_payload = stable_base_json
                txt2img_request_payload["hr_negative_prompt"] = negative_prompt
                txt2img_request_payload["negative_prompt"] = negative_prompt
                txt2img_request_payload["hr_prompt"] = prompt
                txt2img_request_payload["prompt"] = prompt
                txt2img_request_payload["hr_scale"] = quality.value
                
                try:
                    response = json.loads(requests.post(
                        "http://127.0.0.1:7861/sdapi/v1/txt2img", json=txt2img_request_payload
                    ).content.decode("utf8"))

                    with open(f"{os.getcwd()}/BOT/_utils/_tmp/stable_image.png", "wb") as image_file:
                        image_file.write(base64.b64decode(response["images"][0]))
                
                    with open(f"{os.getcwd()}/BOT/_utils/_tmp/stable_image.png", 'rb') as pic:
                        imagine_file = discord.File(pic)
                    
                    negative_part = f"- (negative prompt: {negative_prompt})"
                    
                    await interaction.followup.send(
                        f"**{prompt} {negative_part if negative_prompt != '' else ''}** - <@{interaction.user.id}> (fast)",
                        file=imagine_file,
                        view=ImagineView()
                    )
                    
                    interaction.client._fnbb_globals["imagine_generating"] = False
                    
                except ConnectionError as e:
                    await interaction.followup.send(
                        embed=generic_colored_embed(
                            title="ERROR",
                            description="A connection error with stable-diffusion occured. Perhaps the stable-diffusion API is not running.",
                            footer_usr=interaction.user.global_name,
                            footer_img=interaction.user.avatar,
                            color="ERROR"
                        )
                    )
