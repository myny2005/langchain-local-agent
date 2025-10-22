import requests
from bs4 import BeautifulSoup


def fetch_url(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = " ".join(text.split())
    return text


if __name__ == "__main__":
    example_url = "https://www.theguardian.com/world/2025/oct/22/zelenskyy-calls-trumps-proposal-to-freeze-war-at-current-frontlines-good-compromise"  # random Guardian article
    n = 10000
    print(fetch_url(example_url)[:n])  # show the n chars
