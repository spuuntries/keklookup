from datetime import datetime
from multiprocessing.dummy import Pool
import discord

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
                with tor_requests_session() as s:
                    retries = Retry(
                        total=RETRIES,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504],
                    )
                    s.mount("https://", HTTPAdapter(max_retries=retries))
                    urls = [
                        f"https://yandex.com/images/search?rpt=imageview&cbir_page=similar&url={attachment.url}"
                    ]
                    pool = Pool(2)
