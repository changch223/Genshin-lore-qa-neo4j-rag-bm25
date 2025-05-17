import requests
from bs4 import BeautifulSoup
from typing import List
from langchain.schema import Document

def fetch_wiki_text(url: str) -> str:
    """擷取單頁面所有段落文字"""
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    div = soup.find("div", {"class": "mw-parser-output"})
    if not div:
        return ""
    paras = div.find_all("p")
    return "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))

# Fetch and merge content from multiple wiki URLs
def fetch_multiple_wiki_texts(urls: List[str]) -> str:
    """合併多個 URL 的文字"""
    return "\n\n".join(fetch_wiki_text(u) for u in urls)

# Fetch all Archon Quest page URLs from the Genshin Wiki
def fetch_act_urls() -> List[str]:
    """抓取所有 Archon Quest 頁面的 URL 列表"""
    base = "https://genshin-impact.fandom.com"
    idx_url = f"{base}/wiki/Archon_Quest"
    resp = requests.get(idx_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_=lambda c: c and "article-table" in c)
    if not table:
        return []
    urls = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) >= 2:
            for a in cells[1].find_all("a", href=True):
                href = a["href"]
                if href.startswith("/wiki/"):
                    urls.append(base + href)
    # 去重
    return list(dict.fromkeys(urls))


def fetch_page_text_with_headings(url: str) -> List[Document]:
    """從 Wiki 抓取帶有 H2/H3 標題的段落文本"""
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
                        metadata={"source": url, "section_title": current_heading}
                    )
                )
    return docs


def save_all_archon_quests_to_txt(output_path: str = "archon_quests.txt"):
    """抓取所有 Archon Quest 頁面並存成本地 .txt 檔案"""
    urls = fetch_act_urls()
    print(f"Found {len(urls)} Archon Quest URLs")

    success_count = 0
    fail_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, url in enumerate(urls, 1):
            print(f" ({i}/{len(urls)}) Fetching: {url}")
            try:
                content = fetch_wiki_text(url)
                f.write(f"=== {url} ===\n")
                if content:
                    f.write(content + "\n\n")
                    success_count += 1
                else:
                    f.write("(No content found)\n\n")
            except Exception as e:
                print(f"⚠️ Error on {url}: {e}")
                f.write(f"(Failed to fetch content: {e})\n\n")
                fail_count += 1

    print(f"✅ Completed. Saved to `{output_path}`")
    print(f"✅ Pages fetched successfully: {success_count}")
    print(f"❌ Pages failed: {fail_count}")
