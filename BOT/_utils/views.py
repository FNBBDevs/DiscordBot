import os
import json
import uuid
import discord
import discord.interactions
import datetime
from .embeds import generic_colored_embed, get_imagine_upscale_embed
from .queueing import StableQueueItem


class PauseView(discord.ui.View):
    @discord.ui.button(label="Pause", style=discord.ButtonStyle.primary, emoji="⏸️")
    async def pause_button_callback(self, interaction, button):
        await interaction.response.defer()

        embed = generic_colored_embed(
            title="Success ✅",
            description="Song has been paused",
            color="PURPLE",
        )

        interaction.guild.voice_client.pause()

        original_response = await interaction.original_response()

        await original_response.edit(
            embed=generic_colored_embed(
                title="Success ✅",
                description="Song has been resumed",
                color="PURPLE",
            ),
            view=None,
        )

        await interaction.followup.send(embed=embed, view=ResumeView())


class ResumeView(discord.ui.View):
    @discord.ui.button(label="Resume", style=discord.ButtonStyle.primary, emoji="▶️")
    async def resume_button_callback(self, interaction, button):
        await interaction.response.defer()
        embed = generic_colored_embed(
            title="Success ✅",
            description="Song has been resumed",
            color="PURPLE",
        )
        interaction.guild.voice_client.resume()

        original_response = await interaction.original_response()

        await original_response.edit(
            embed=generic_colored_embed(
                title="Success ✅",
                description="Song has been paused",
                color="PURPLE",
            ),
            view=None,
        )

        await interaction.followup.send(embed=embed)


class PlayingView(discord.ui.View):
    @discord.ui.button(label="Pause", style=discord.ButtonStyle.primary, emoji="⏸️")
    async def pause_button_callback(self, interaction, button):
        await interaction.response.defer()

        embed = generic_colored_embed(
            title="Success ✅",
            description="Song has been paused",
            color="PURPLE",
        )

        interaction.guild.voice_client.pause()

        original_response = await interaction.original_response()

        await original_response.edit(embed=original_response.embeds[0], view=None)

        await interaction.followup.send(
            embed=generic_colored_embed(
                title="Success ✅",
                description="Song has been paused",
                color="PURPLE",
            ),
            view=ResumeView(),
        )

    @discord.ui.button(label="Skip", style=discord.ButtonStyle.danger, emoji="⏭️")
    async def skip_button_callback(self, interaction, button):
        user_channel = interaction.user.voice
        voice_channel = interaction.guild.voice_client

        # AWAIT A RESPONSE
        await interaction.response.defer()

        original_response = await interaction.original_response()

        await original_response.edit(embed=original_response.embeds[0], view=None)

        # IF THE USER IS NOT THE VOICE CHANNEL, ISSUE ERROR
        if not user_channel:
            await interaction.followup.send(
                embed=generic_colored_embed(
                    title="You are Not In a Voice Channel!",
                    description=("Hermph... ❌"),
                    footer_text="(Attempted) Skip By:",
                    footer_usr=interaction.user.name,
                    footer_img=interaction.user.guild_avatar,
                    color="ERROR",
                )
            )

        # SEE IF THE BOT IS IN THE CHANNEL
        elif voice_channel:
            # If the bot is currently playing music
            if voice_channel.is_playing():
                # This is BROKEN
                voice_channel.stop()

                await interaction.followup.send(
                    embed=generic_colored_embed(
                        title="Successfully Skipped Song!",
                        description="",
                        footer_text="Skipped By:",
                        footer_usr=interaction.user.name,
                        footer_img=interaction.user.guild_avatar,
                        color="SUCCESS",
                    )
                )
            # If the bot is in the voice channel but no music is playing
            else:
                await interaction.followup.send(
                    embed=generic_colored_embed(
                        title="No Song to Skip!",
                        description=(
                            "Erm... The queue is empty and there is no song to"
                            " skip..."
                        ),
                        footer_text="(Attempted) Skip By:",
                        footer_usr=interaction.user.name,
                        footer_img=interaction.user.guild_avatar,
                        color="WARNING",
                    )
                )

        # BOT IS NOT IN THE CHANNEL
        else:
            await interaction.followup.send(
                embed=generic_colored_embed(
                    title="Bot Not In Voice Channel!",
                    description=(
                        "Hermph... I must be in a voice channel to skip songs..."
                    ),
                    footer_text="(Attempted) Skip By:",
                    footer_usr=interaction.user.name,
                    footer_img=interaction.user.guild_avatar,
                    color="ERROR",
                )
            )


class ImagineView(discord.ui.View):    
    def __init__(
        self,
        stable_id,
        prompt,
        negative,
        quality,
        cfg,
        steps,
        seed,
        upscale_model,
        sampler,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, timeout=None)
        self.stable_id = stable_id
        self.prompt = prompt
        self.negative_prompt = negative
        self.quality = quality
        self.cfg_scale = cfg
        self.steps = steps
        self.seed = seed
        self.upscale_model = upscale_model
        self.sampler = sampler
            
    @discord.ui.button(
        label="U1", style=discord.ButtonStyle.gray, row=1, custom_id="id_U1"
    )
    async def u1(self, interaction: discord.Interaction, button: discord.ui.Button):

        print(f"U1 clicked with stable id: {self.stable_id}")

        # disable the button and make it blue
        button.disabled = True
        button.style = discord.ButtonStyle.primary

        # create the embed that will contain the single image (determined by index)
        embed, file = get_imagine_upscale_embed(
            index=1,
            footer_text="Upscale requested by: ",
            footer_usr=interaction.user.global_name,
            footer_img=interaction.user.avatar,
            stable_id=self.stable_id,
        )

        # edit the view to the button is disabled on the ui side
        await interaction.response.edit_message(view=self)
        # send the single image
        if file:
            await interaction.followup.send(embed=embed, file=file)
        else:
            await interaction.followup.send(embed=embed)

    @discord.ui.button(
        label="U2", style=discord.ButtonStyle.gray, row=1, custom_id="id_U2"
    )
    async def u2(self, interaction, button):

        button.disabled = True
        button.style = discord.ButtonStyle.primary

        embed, file = get_imagine_upscale_embed(
            index=2,
            footer_text="Upscale requested by: ",
            footer_usr=interaction.user.global_name,
            footer_img=interaction.user.avatar,
            stable_id=self.stable_id,
        )

        await interaction.response.edit_message(view=self)
        if file:
            await interaction.followup.send(embed=embed, file=file)
        else:
            await interaction.followup.send(embed=embed)

    @discord.ui.button(
        label="U3", style=discord.ButtonStyle.gray, row=1, custom_id="id_U3"
    )
    async def u3(self, interaction, button):

        button.disabled = True
        button.style = discord.ButtonStyle.primary

        embed, file = get_imagine_upscale_embed(
            index=3,
            footer_text="Upscale requested by: ",
            footer_usr=interaction.user.global_name,
            footer_img=interaction.user.avatar,
            stable_id=self.stable_id,
        )

        await interaction.response.edit_message(view=self)
        if file:
            await interaction.followup.send(embed=embed, file=file)
        else:
            await interaction.followup.send(embed=embed)

    @discord.ui.button(
        label="U4", style=discord.ButtonStyle.gray, row=1, custom_id="id_U4"
    )
    async def u4(self, interaction, button):

        button.disabled = True
        button.style = discord.ButtonStyle.primary

        embed, file = get_imagine_upscale_embed(
            index=4,
            footer_text="Upscale requested by: ",
            footer_usr=interaction.user.global_name,
            footer_img=interaction.user.avatar,
            stable_id=self.stable_id,
        )

        await interaction.response.edit_message(view=self)
        if file:
            await interaction.followup.send(embed=embed, file=file)
        else:
            await interaction.followup.send(embed=embed)

    @discord.ui.button(label="", style=discord.ButtonStyle.gray, emoji="🔁", row=1)
    async def redo(self, interaction, button):
        
        button.disabled = True
        button.style = discord.ButtonStyle.primary
        
        with open(f"{os.getcwd()}\\BOT\\_utils\\_tmp\\stable_diffusion\\{self.stable_id}\\info.json") as json_file:
            json_info = json.loads(json_file.read())
        
        queue_item = StableQueueItem().from_dict(json_info)
        
        new_id = uuid.uuid4().hex
        while new_id in os.listdir(f"{os.getcwd()}\\BOT\\_utils\\_tmp\\stable_diffusion"):
            new_id = uuid.uuid4().hex
        queue_item.stable_id = new_id
        queue_item.user = interaction.user.global_name
        queue_item.user_avatar = interaction.user.avatar

        await interaction.response.edit_message(view=self)
        await interaction.client._fnbb_globals["SCC"].delegate(queue_item)

    @discord.ui.button(label="V1", style=discord.ButtonStyle.gray, row=2)
    async def v1(self, interaction, button):
        await interaction.response.send_message("balls", ephemeral=True)

    @discord.ui.button(label="V2", style=discord.ButtonStyle.gray, row=2)
    async def v2(self, interaction, button):
        await interaction.response.send_message("balls", ephemeral=True)

    @discord.ui.button(label="V3", style=discord.ButtonStyle.gray, row=2)
    async def v3(self, interaction, button):
        await interaction.response.send_message("balls", ephemeral=True)

    @discord.ui.button(label="V4", style=discord.ButtonStyle.gray, row=2)
    async def v4(self, interaction, button):
        await interaction.response.send_message("balls", ephemeral=True)

    @discord.ui.button(label="", style=discord.ButtonStyle.gray, emoji="ℹ️", row=2)
    async def info(self, interaction, button):
        embed = discord.Embed(
            title=f"✨ Info for {self.stable_id} ✨",
            description="Here is the info for the image generation!",
            color=0x333333,
            timestamp=datetime.datetime.now(),
        )
        embed.add_field(name="Prompt", value=self.prompt, inline=False)
        embed.add_field(name="Negative Prompt", value=self.negative_prompt, inline=False)
        embed.add_field(name="Upscale", value=self.quality, inline=False)
        embed.add_field(name="CFG Scale", value=self.cfg_scale, inline=False)
        embed.add_field(name="Steps", value=self.steps, inline=False)
        embed.add_field(name="Seed", value=self.seed, inline=False)
        embed.add_field(name="Upscale Model", value=self.upscale_model, inline=False)
        embed.add_field(name="Sampler", value=self.sampler, inline=False)
        embed.set_footer(text=f"Requested by {interaction.user.global_name}", icon_url=interaction.user.avatar)
        
        button.disabled = True
        
        await interaction.response.edit_message(view=self)
        await interaction.followup.send(embed=embed)
