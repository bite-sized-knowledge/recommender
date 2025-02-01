import re
from bs4 import BeautifulSoup

def clean_html(html: str, blog_id:int) -> str:
    print("CLEANING HTML")
    """
    입력된 HTML에서 본문 텍스트만 추출하고 필요한 전처리를 수행
    """
    html = html.lower() 
    html = re.sub(r"'", "", html)
    html = re.sub(r'[“”]', '"', html)

    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F"  # 이모지
        r"\U0001F300-\U0001F5FF"  # 기호 및 픽토그램
        r"\U0001F680-\U0001F6FF"  # 운송 및 기계
        r"\U0001F700-\U0001F77F]"  # 추가 범위
        , re.UNICODE
    )
    html = emoji_pattern.sub("", html)

    # 1. HTML 파싱
    soup = BeautifulSoup(html, "lxml")

    # 2. 불필요한 태그 제거
    components = ["style", "meta", "link", "head", "noscript", "iframe", "form", "footer", "header", "nav", "aside","img", "script"]
    for tag in soup(components):
        tag.decompose()

    # 3. 코드 블록 유지 (pre 태그 변환)
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        lang = pre.get("class", ["plaintext"])[0]  # 언어 감지 (기본값: plaintext)
        code_content = code.get_text().strip() if code else pre.get_text().strip()
        pre.replace_with(f"<code language='{lang}'>\n{code_content}\n</code>")
     
    # 4. 댓글 섹션 제거 (Disqus, Utterances 등)
    comment_patterns = [
        re.compile(r'\bdisqus\b', re.I),
        re.compile(r'\bgisqus\b', re.I),
        re.compile(r'\butterances\b', re.I)
    ]
    for div in soup.find_all("div"):
        if any(pattern.search(str(div)) for pattern in comment_patterns):
            div.decompose()

    if blog_id == 4: return html
    return soup.get_text()

class BlogPostProcessor:
    def __init__(self, text, blog_id):
        self.text = clean_html(text, blog_id)
        self.text = re.sub(r'\n{2,}', '', self.text)
        self.text = re.sub(r'\s{2,}', ' ', self.text)
        self.text = re.sub(r'\s+([.,!?])', r'\1', self.text)
        self.text = re.sub(r'\'', '', self.text)
        self.text = re.sub(r'”', '"', self.text)

    def process(self):
        raise NotImplementedError("Subclasses should implement this method")
