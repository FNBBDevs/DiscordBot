import os
import json
import time
import openai
import base64
import typing
import discord
import asyncio
import requests
import functools

from PIL import Image

from .views import ImagineView
from .embeds import get_imagine_embed
from .stable_diffusion import stable_base_json
from .queueing import StableQueue, StableQueueItem


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


class StableCommandCenter:
    def __init__(self, current_directory: str, client: discord.Client):
        self.current_directory = f"{current_directory}\\BOT\\_utils\\_tmp\\stable_diffusion"
        self.client = client
        self.queue = StableQueue()
        self.processing = False
        self.queue_empty = True

    async def delegate(self, stable_request: StableQueueItem):
        # add to queue if processing
        if self.processing:
            await self.add_to_queue(stable_request)
        else:
            # process the request
            self.processing = True
            embed, file, view, channel  = await self.process_stable_queue_item(item=stable_request)
    
            await channel.send(
                embed=embed, 
                file=file,
                view=view
            )
                        
            await self.check_processing()
            
            

    async def add_to_queue(self, item: StableQueueItem):
        self.queue_empty = False
        self.queue.add(item)

    async def check_processing(self):
        if len(self.queue) > 0:
            await self.process_queue()
        else:
            self.processing = False

    @to_thread
    def call_txt2img(self, payload: dict):
        response = json.loads(
            requests.post(
                url=r"http://127.0.0.1:7861/sdapi/v1/txt2img", json=payload
            ).content.decode("utf8")
        )
        return response

    def save_image(self, path: str, data: str):
        with open(
            path,
            "wb",
        ) as image_file:
            image_file.write(base64.b64decode(data))

    def make_grid_image(self, stable_id: str):
        # create a 2x2 grid of the images to send (like mid journey)
        pil_images = []
        for image_file in os.listdir(f"{self.current_directory}\\{stable_id}"):
            if "grid" not in image_file and "json" not in image_file:
                pil_images.append(
                    Image.open(f"{self.current_directory}\\{stable_id}\\{image_file}")
                )

        grid_image = Image.new(
            "RGB", (pil_images[0].width * 2, pil_images[0].height * 2)
        )

        pil_images_count = len(pil_images)

        if pil_images_count >= 1:
            grid_image.paste(pil_images[0], (0, 0))
        if pil_images_count >= 2:
            grid_image.paste(pil_images[1], (pil_images[0].width, 0))
        if pil_images_count >= 3:
            grid_image.paste(pil_images[2], (0, pil_images[0].height))
        if pil_images_count >= 4:
            grid_image.paste(pil_images[3], (pil_images[0].width, pil_images[0].height))

        grid_image = grid_image.resize(
            (pil_images[0].width // 2, pil_images[0].height // 2),
            Image.Resampling.BICUBIC,
        )

        grid_image.save(f"{self.current_directory}\\{stable_id}\\{stable_id}_grid.png")

    async def process_queue(self):
        while len(self.queue) != 0:
            stable_queue_item = self.queue.pop()
            embed, file, view, channel = await self.process_stable_queue_item(item=stable_queue_item)
            await channel.send(
                embed=embed, 
                file=file,
                view=view
            )
        self.processing = False
        self.queue_empty = True

    async def process_stable_queue_item(self, item: StableQueueItem):
        user = item.user
        user_avatar = item.user_avatar
        channel = item.channel
        stable_id = item.stable_id
        source_channel = self.client.get_channel(channel)
        
        os.mkdir(f"{self.current_directory}\\{stable_id}")
        
        with open(f"{self.current_directory}\\{stable_id}\\info.json", "w") as json_file:
            info_json = {
                "prompt": item.prompt,
                "negative_prompt": item.negative_prompt,
                "quality": item.quality,
                "cfg_scale": item.cfg_scale,
                "steps": item.steps,
                "seed": item.seed,
                "upscale_model": item.upscale_model,
                "sampler": item.sampler,
                "channel": item.channel,
                "stable_id": stable_id,
                "user": item.user,
                "user_avatar": item.user_avatar if isinstance(item.user_avatar, str) else item.user_avatar.url,
                "images": item.images
            }
            
            json_file.write(json.dumps(info_json, indent=3))

        txt2img_request_payload = stable_base_json.copy()
        txt2img_request_payload["hr_negative_prompt"] = item.negative_prompt
        txt2img_request_payload["negative_prompt"] = item.negative_prompt
        txt2img_request_payload["hr_prompt"] = item.prompt
        txt2img_request_payload["prompt"] = item.prompt
        txt2img_request_payload["hr_scale"] = item.quality
        txt2img_request_payload["cfg_scale"] = item.cfg_scale
        txt2img_request_payload["steps"] = item.steps
        txt2img_request_payload["seed"] = item.seed
        txt2img_request_payload["hr_upscaler"] = item.upscale_model
        txt2img_request_payload["n_iter"] = item.images
        if item.sampler:
            txt2img_request_payload["sampler_name"] = item.sampler

        response = await self.call_txt2img(payload=txt2img_request_payload)

        for idx, image in enumerate(response["images"]):
            self.save_image(
                path=f"{self.current_directory}\\{stable_id}\\{stable_id}_{idx}.png",
                data=image,
            )

        self.make_grid_image(stable_id=stable_id)

        imagine_emebed, imagine_image_file = get_imagine_embed(
            footer_text="Image generated by: ",
            footer_usr=user,
            footer_img=user_avatar,
            stable_id=stable_id,
        )
        imagine_view = ImagineView(
                stable_id=stable_id,
                prompt=item.prompt,
                negative=item.negative_prompt,
                quality=item.quality,
                cfg=item.cfg_scale,
                steps=item.steps,
                seed=item.seed,
                upscale_model=item.upscale_model,
                sampler=item.sampler,
            )
        
        return imagine_emebed, imagine_image_file, imagine_view, source_channel
