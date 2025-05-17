# data_extraction.py

import requests
from bs4 import BeautifulSoup
from typing import List
from langchain.schema import Document

# Fetch all Archon Quest page URLs from the Genshin Wiki
def fetch_act_urls() -> List[str]:
    """
    爬取 Archon Quest 总览页中所有分支任务链接，返回去重后的完整 URL 列表。
    """
    url = "https://genshin-impact.fandom.com/wiki/Archon_Quest"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_=lambda c: c and "article-table" in c)
    urls = []
    if not table:
        return urls

    # Loop through the table rows to find quest links
    for row in table.find_all("tr")[1:]:  # Skip header
        cells = row.find_all("td")
        if len(cells) >= 2:
            for a in cells[1].find_all("a", href=True):
                href = a["href"]
                if href.startswith("/wiki/"):
                    full = "https://genshin-impact.fandom.com" + href
                    urls.append(full)
    return list(dict.fromkeys(urls))  # Remove duplicates


# Web scraping function to extract main story text
def fetch_wiki_text(url):
    """
    请求指定 URL，返回页面中 <div class="mw-parser-output"> 内所有 <p> 标签拼接后的纯文本。
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    paras = soup.find("div", {"class": "mw-parser-output"}).find_all("p")
    return "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))

# Fetch and merge content from multiple wiki URLs
def fetch_multiple_wiki_texts(urls: List[str]) -> str:
    """
    将多个 URL 的文本依次 fetch_wiki_text，再以双换行拼接。
    """
    return "\n\n".join(fetch_wiki_text(url) for url in urls)

# Scrape Archon Quest URLs & Extract Page Text with Headings
def fetch_page_text_with_headings(url: str) -> List[Document]:
    """
    Extracts <p> text blocks from a given Genshin Wiki page and assigns each to the most recent <h2> or <h3> heading.
    Returns a list of LangChain Document objects with section titles as metadata.
    """
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    div = soup.find("div", class_="mw-parser-output")
    docs = []
    current_heading = ""
    if not div:
        return docs
    for elem in div.children:
        if elem.name in ["h2", "h3"]:
            current_heading = elem.get_text(strip=True)
        elif elem.name == "p":
            text = elem.get_text(strip=True)
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "section_title": current_heading
                        }
                    )
                )
    return docs