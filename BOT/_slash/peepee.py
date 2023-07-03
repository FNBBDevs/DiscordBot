import random

import discord

_MESSAGES = [
    "big and chunky dick",
    "non-existant dick like Koi Carpe",
    "tiny dick",
    "uncircumcised penis",
    "some dick cheese",
    "an inny",
    "a chode",
    "erecile dysfunction (dead willie)",
    "a priapism",
    "beautiful penis",
    "sideways dong",
    "average dick",
    "millimeter peter",
    "1-inch finch",
    "decapitated ding-dong",
    "long dragon",
    "cock flock",
    "**T H I C C** dick",
    "dick magnifique",
    "high-IQ wiener",
    "neutered pecker",
    "abuser bruiser",
    "booster rooster",
    "strong schlong",
    "walking stocking",
]

_ICONS = [
    "https://pbs.twimg.com/profile_images/480552447186714626/TJqvzNu4.jpeg",
    "https://mysahana.org/wp-content/uploads/2010/09/eggplant.jpg",
    "https://www.kindpng.com/picc/m/127-1272622_downloads-12-eggplant-royalty-free-clipart-eggplant-with.png",
    "https://thesportsgrail.com/wp-content/uploads/2022/08/Andrew-tate-biography-business-family-brother-mother-cars-net-worth-father-age-nationality-height-boxing.jpg",
    "https://theacademyadvocate.com/wp-content/uploads/2022/09/AndrewTate.png",
    "https://i.etsystatic.com/24260052/r/il/9e90fd/3090533131/il_fullxfull.3090533131_qrip.jpg",
]


class Peepee:
    def __init__(self, tree, guild):
        @tree.command(
            name="pp", description="Slimey and Gooey", guild=discord.Object(id=guild)
        )
        async def pp(interaction: discord.Interaction, user: discord.User = None):
            if not user:
                # Default to the user who sent the interaction
                user = interaction.user

            embed = discord.Embed(
                title="What the DICK???",
                description=f"{user.mention} has {random.choice(_MESSAGES)}",
                color=0x5CB3FF,
            )

            embed.set_author(
                name="Weenus Peenus.com",
                url="https://nulzo.github.io/",
                icon_url=random.choice(_ICONS),
            )
            embed.set_thumbnail(url=user.display_avatar.url)

            await interaction.response.send_message(content=user.mention, embed=embed)
            await interaction.edit_original_response(content="\b")
