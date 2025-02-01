import re
from .base import BlogPostProcessor
from bs4 import BeautifulSoup
from .unicode_escape import decode_unicode_escapes


class KakaoProcessor(BlogPostProcessor):
    def _preprocess_to_html(self):
        # 카카오는 원문이 script 태그 안에 있고 unescape도 필요하기에 처리 필요
        html = decode_unicode_escapes(self.text)
    
        soup = BeautifulSoup(html, 'lxml')
        script_text = soup.find('script', {'id': '__NUXT_DATA__'})
        if script_text is None:
            return html
        return script_text.prettify()

    def process(self):
        self.text = self._preprocess_to_html()

        match = re.search(r'(<p>.*)', self.text)
        self.text = match.group(1).strip()
        self.text = re.sub(r'<.*?>', '', self.text)
        self.text = self.text.replace('\\n', '\n')
        self.text = re.sub(r'\n{3,}', '\n\n', self.text)
        self.text = re.sub(r'\s{2,}', ' ', self.text)
        self.text = re.sub(r'\s+([.,!?])', r'\1', self.text)
        self.text = re.sub(r"'", "", self.text)
        self.text = self.text.split("함께 하면 좋은 글")[0]
        self.text = self.text.split('{"id"')[0]
        return self.text.strip()