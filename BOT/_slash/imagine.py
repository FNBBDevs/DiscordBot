import os
import discord
from discordwebhook import Discord
from _utils.views import ImagineView
import io
import json
import base64
import requests

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
        async def imagine(interaction: discord.Interaction, prompt: str, negative_prompt: str = ""):
            """
            /imagine
            """
            
            await interaction.response.defer()

            txt2img_request_payload = {
                "alwayson_scripts": {
                    "API payload": {"args": []},
                    "Additional networks for generating": {
                        "args": [
                            False,
                            False,
                            "LoRA",
                            "None",
                            0,
                            0,
                            "LoRA",
                            "None",
                            0,
                            0,
                            "LoRA",
                            "None",
                            0,
                            0,
                            "LoRA",
                            "None",
                            0,
                            0,
                            "LoRA",
                            "None",
                            0,
                            0,
                            None,
                            "Refresh models",
                        ]
                    },
                    "Dynamic Prompts v2.17.1": {
                        "args": [
                            True,
                            False,
                            1,
                            False,
                            False,
                            False,
                            1.1,
                            1.5,
                            100,
                            0.7,
                            False,
                            False,
                            True,
                            False,
                            False,
                            0,
                            "Gustavosta/MagicPrompt-Stable-Diffusion",
                            "",
                        ]
                    },
                    "Extra options": {"args": []},
                    "Hypertile": {"args": []},
                    "Refiner": {"args": [False, "", 0.8]},
                    "Seed": {"args": [-1, False, -1, 0, 0, 0]},
                },
                "batch_size": 1,
                "cfg_scale": 3,
                "comments": {},
                "denoising_strength": 0.7,
                "disable_extra_networks": False,
                "do_not_save_grid": False,
                "do_not_save_samples": False,
                "enable_hr": True,
                "height": 480,
                "hr_negative_prompt": negative_prompt if negative_prompt else "",
                "hr_prompt": prompt if prompt else "",
                "hr_resize_x": 0,
                "hr_resize_y": 0,
                "hr_scale": 2,
                "hr_second_pass_steps": 0,
                "hr_upscaler": "Latent",
                "n_iter": 1,
                "negative_prompt": negative_prompt if negative_prompt else "",
                "override_settings": {},
                "override_settings_restore_afterwards": True,
                "prompt": prompt if prompt else "",
                "restore_faces": False,
                "s_churn": 0.0,
                "s_min_uncond": 0.0,
                "s_noise": 1.0,
                "s_tmax": None,
                "s_tmin": 0.0,
                "sampler_name": "DDIM",
                "script_args": [],
                "script_name": None,
                "seed": -1,
                "seed_enable_extras": True,
                "seed_resize_from_h": -1,
                "seed_resize_from_w": -1,
                "steps": 20,
                "styles": [],
                "subseed": 2408667576,
                "subseed_strength": 0,
                "tiling": False,
                "width": 800,
            }
            
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
