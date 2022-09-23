from typing import Optional, TypedDict
from torpy.http.requests import do_request
import multiprocessing as mp
from bs4 import BeautifulSoup
import urllib.parse


class similarimages(TypedDict):
    url: str
    sim: int


async def lookup(image: str, threshold: Optional[int]) -> list[similarimages]:
    res = do_request(
        url=f"https://yandex.com/images/search?rpt=imageview&url={urllib.parse.quote_plus(image)}",
        retries=3,
    )
    html = BeautifulSoup(res)
    similars = html.find_all("a", class_="Thumb_type_inline")
    for similar in similars[0:15]:
        sim_image = similar["href"]
