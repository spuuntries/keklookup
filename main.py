from datetime import datetime
import discord
from lookup import lookup

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


def log(message: str):
    print(f"[{datetime.now().isoformat()}] {message}")


@client.event
async def on_ready():
    log(f"Logged in as {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    if len(message.attachments) > 0:
        for attachment in message.attachments:
            if attachment.content_type in ["image/jpeg", "image/png", "image/avif"]:
                similars = await lookup(attachment.url)
                if len(similars) < 0:
                    return
                for sim in similars: 
                    sim[]
