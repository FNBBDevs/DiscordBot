import os
import json
import base64
import typing
import discord
import asyncio
import requests
import functools

from PIL import Image
from enum import Enum

from .embeds import get_imagine_embed
from .views import ImagineView
from .queueing import StableQueue

# upscale options
class Upscale(Enum):
    one = 1
    two = 2
    three = 3


class UpscaleModel(Enum):
    none = "None"
    lanczos = "Lanczos"
    latent = "Latent"
    latent_antialiased = "Latent (antialiased)"
    latent_bicubic = "Latent (bicubic)"
    latent_bicubic_antialiased = "Latent (bicubic antialiased)"
    latent_nearest = "Latent (nearest)"
    latent_nearest_exact = "Latent (nearest-exact)"
    nearest = "Nearest"
    esrgan_4x = "ESRGAN_4x"
    ldsr = "LDSR"
    r_esrgan_4x = "R-ESRGAN 4x+"
    r_esrgan_4x_anime6b = "R-ESRGAN 4x+ Anime6B"
    scunet_gan = "ScuNET GAN"
    scunet_psnr = "ScuNET PSNR"
    swinir_4x = "SwinIR 4x"


class SamplerSetTwo(Enum):
    none = None
    dpm_2m_karras = "DPM++ 2M Karras"
    dpm_sde_karras = "DPM++ SDE Karras"
    dpm_2m_sde_exponential = "DPM++ 2M SDE Exponential"
    dpm_2m_sde_karras = "DPM++ 2M SDE Karras"
    euler_a = "Euler a"
    euler = "Euler"
    lms = "LMS"
    heun = "Heun"
    dpm2 = "DPM2"
    dpm2_a = "DPM2 a"
    dpm_2s_a = "DPM++ 2S a"
    dpm_2m = "DPM++ 2M"
    dpm_sde = "DPM++ SDE"
    dpm_2m_sde = "DPM++ 2M SDE"
    dpm_2m_sde_heun = "DPM++ 2M SDE Heun"
    dpm_2m_sde_heun_karras = "DPM++ 2M SDE Heun Karras"


class SamplerSetOne(Enum):
    none = None
    dpm_2m_sde_heun_exponential = "DPM++ 2M SDE Heun Exponential"
    dpm_3m_sde = "DPM++ 3M SDE"
    dpm_3m_sde_karras = "DPM++ 3M SDE Karras"
    dpm_3m_sde_exponential = "DPM++ 3M SDE Exponential"
    dpm_fast = "DPM fast"
    dpm_adaptive = "DPM adaptive"
    lms_karras = "LMS Karras"
    dpm2_karras = "DPM2 Karras"
    dpm2_a_karras = "DPM2 a Karras"
    dpm_2s_a_karras = "DPM++ 2S a Karras"
    restart = "Restart"
    ddim = "DDIM"
    plms = "PLMS"
    unipc = "UniPC"


# payload to send to txt2img
stable_base_json = {
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
    "batch_count": 4,
    "cfg_scale": 3,
    "comments": {},
    "denoising_strength": 0.7,
    "disable_extra_networks": False,
    "do_not_save_grid": False,
    "do_not_save_samples": False,
    "enable_hr": True,
    "height": 480,
    "hr_negative_prompt": "",
    "hr_prompt": "",
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_scale": 3,
    "hr_second_pass_steps": 0,
    "hr_upscaler": "Latent",
    "n_iter": 4,
    "negative_prompt": "",
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "prompt": "",
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


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


@to_thread
def call_txt2img(payload):
    response = json.loads(
        requests.post(
            url=r"http://127.0.0.1:7861/sdapi/v1/txt2img", json=payload
        ).content.decode("utf8")
    )

    return response


def save_image(path: str, data: str):
    with open(
        path,
        "wb",
    ) as image_file:
        image_file.write(base64.b64decode(data))


def make_grid_image(stable_id: str):
    # create a 2x2 grid of the images to send (like mid journey)
    pil_images = []
    for image_file in os.listdir(f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/{stable_id}"):
        if "grid" not in image_file:
            pil_images.append(
                Image.open(
                    f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/{stable_id}/{image_file}"
                )
            )

    grid_image = Image.new("RGB", (pil_images[0].width * 2, pil_images[0].height * 2))

    grid_image.paste(pil_images[0], (0, 0))
    grid_image.paste(pil_images[1], (pil_images[0].width, 0))
    grid_image.paste(pil_images[2], (0, pil_images[0].height))
    grid_image.paste(pil_images[3], (pil_images[0].width, pil_images[0].height))

    grid_image = grid_image.resize(
        (pil_images[0].width // 2, pil_images[0].height // 2),
        Image.Resampling.BICUBIC,
    )

    grid_image.save(f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/{stable_id}/{stable_id}_grid.png")


async def process_queue(client: discord.Client):
    queue_length = len(client._fnbb_globals.get("imagine_queue"))
    while queue_length != 0:
        try:
            stable_queue_item = client._fnbb_globals.get("imagine_queue").pop()
           
            user = stable_queue_item.user
            user_avatar = stable_queue_item.user_avatar
            channel = stable_queue_item.channel
            stable_id = stable_queue_item.stable_id

            print(user)
            print(type(user))
            
            source_channel = client.get_channel(channel)

            txt2img_request_payload = stable_base_json.copy()
            txt2img_request_payload["hr_negative_prompt"] = stable_queue_item.negative_prompt
            txt2img_request_payload["negative_prompt"] = stable_queue_item.negative_prompt
            txt2img_request_payload["hr_prompt"] = stable_queue_item.prompt
            txt2img_request_payload["prompt"] = stable_queue_item.prompt
            txt2img_request_payload["hr_scale"] = stable_queue_item.quality
            txt2img_request_payload["cfg_scale"] = stable_queue_item.cfg_scale
            txt2img_request_payload["steps"] = stable_queue_item.steps
            txt2img_request_payload["seed"] = stable_queue_item.seed
            txt2img_request_payload["hr_upscaler"] = stable_queue_item.upscale_model
            if stable_queue_item.sampler:
                txt2img_request_payload["sampler_name"] = stable_queue_item.sampler
            else:
                # send none sampler error message
                break

            # ping local hosted stable diffusion ai
            response = await call_txt2img(payload=txt2img_request_payload)

            # save all 4 images
            for idx, image in enumerate(response["images"]):
                save_image(
                    path=f"{os.getcwd()}/BOT/_utils/_tmp/stable_diffusion/{stable_id}/{stable_id}_{idx}.png",
                    data=image
                )

            make_grid_image(stable_id=stable_id)

            imagine_emebed, imagine_image_file = get_imagine_embed(
                prompt=stable_queue_item.prompt,
                negative=stable_queue_item.negative_prompt,
                quality=stable_queue_item.quality,
                cfg=stable_queue_item.cfg_scale,
                steps=stable_queue_item.steps,
                seed=stable_queue_item.seed,
                footer_text="Image generated by: ",
                footer_usr=user,
                footer_img=user_avatar,
                stable_id=stable_id
            )

            await source_channel.send(
                view=ImagineView(stable_id=stable_id),
                embed=imagine_emebed,
                file=imagine_image_file,
            )
            queue_length -= 1
        except Exception as e:
            print(f"ERROR when processing stable queue: {e}")
            client._fnbb_globals["imagine_queue"] = StableQueue()
            client._fnbb_globals["imagine_generating"] = False
            break
    client._fnbb_globals["imagine_queue"] = StableQueue()  
    client._fnbb_globals["imagine_generating"] = False
    

