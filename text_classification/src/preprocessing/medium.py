import re
from .base import BlogPostProcessor

class MediumProcessor(BlogPostProcessor):
    def process(self):
        self.text = re.split(r"--\d+share", self.text)[-1]
        self.text = re.split('published in', self.text)[0]

        return self.text
