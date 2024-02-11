import os
import discord
from discordwebhook import Discord
from _utils.views import ImagineView
from _utils.embeds import generic_colored_embed
from _utils.embeds import imagine as imagine_embed
from _utils.stable_diffusion import stable_base_json, Upscale
import io
import json
import base64
import requests
from requests.exceptions import *
from _utils.alerts import DateTimeAlert
from PIL import Image


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

            # is stable busy? (doesn't work since the api call is blocking)
            if interaction.client._fnbb_globals.get("imagine_generating"):

                await interaction.followup.send(
                    embed=generic_colored_embed(
                        title="Cannot Generate Image",
                        description="An image is already in the process of generating. Add prompts to the queue until stable diffusion frees up.",
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
                        title="Generating!",
                        description="Your image is being generated! It will be sent shortly. (Estimated wait: 1min)",
                        footer_usr=interaction.user.global_name,
                        footer_img=interaction.user.avatar,
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

                try:
                    # ping local hosted stable diffusion ai
                    response = json.loads(
                        requests.post(
                            "http://127.0.0.1:7861/sdapi/v1/txt2img",
                            json=txt2img_request_payload,
                        ).content.decode("utf8")
                    )
                    
                    # save all 4 images
                    for idx, image in enumerate(response["images"]):
                        with open(
                            f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/stable_image_{idx}.png", "wb"
                        ) as image_file:
                            image_file.write(base64.b64decode(image))
                    
                    # create a 2x2 grid of the images to send (like mid journey)
                    pil_images = []
                    for image_file in os.listdir(f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/"):
                        if "grid" not in image_file:
                            pil_images.append(Image.open(f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/{image_file}"))
                    
                    grid_image = Image.new("RGB", (pil_images[0].width * 2, pil_images[0].height * 2))

                    grid_image.paste(pil_images[0], (0,0))
                    grid_image.paste(pil_images[1], (pil_images[0].width,0))
                    grid_image.paste(pil_images[2], (0,pil_images[0].height))
                    grid_image.paste(pil_images[3], (pil_images[0].width,pil_images[0].height))
                    
                    grid_image = grid_image.resize((pil_images[0].width // 3, pil_images[0].height // 3), Image.Resampling.BICUBIC)
                    
                    grid_image.save(f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/grid.png")

                    # send the grid view with buttons to upscale each picture
                    ie, imf = imagine_embed(
                        prompt=prompt,
                        negative=negative_prompt,
                        quality=quality,
                        cfg=cfg_scale,
                        steps=steps,
                        seed=seed,
                        footer_text="Image generated by: ",
                        footer_usr=interaction.user.global_name,
                        footer_img=interaction.user.avatar
                    )

                    await interaction.followup.send(
                        # f"**{prompt} {negative_part if negative_prompt != '' else ''}** - <@{interaction.user.id}> (fast)",
                        view=ImagineView(),
                        embed=ie,
                        file=imf
                    )

                    interaction.client._fnbb_globals["imagine_generating"] = False

                except ConnectionError as e:
                    interaction.client._fnbb_globals["imagine_generating"] = False
                    print(DateTimeAlert(text=f"STABLE DIFFUSION ERROR: {e}", dtia_alert_type="ERROR", message_from="bot._slash.imagine"))
                    await interaction.followup.send(
                        embed=generic_colored_embed(
                            title="ERROR",
                            description="A connection error with stable-diffusion occured. Perhaps the stable-diffusion API is not running.",
                            footer_usr=interaction.user.global_name,
                            footer_img=interaction.user.avatar,
                            color="ERROR",
                        )
                    )
