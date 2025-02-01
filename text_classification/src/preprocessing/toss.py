import re
from .base import BlogPostProcessor

class TossProcessor(BlogPostProcessor):
    def process(self):
        self.text = re.split('관련 문의: toss-tech@toss.im', self.text)[0]
        return self.text