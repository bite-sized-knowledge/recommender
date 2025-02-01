import re
from .base import BlogPostProcessor

class WoowahanProcessor(BlogPostProcessor):
    def process(self):
        self.text = re.split('{{sub.name}}', self.text)[-1]
        self.text = re.split('leave a comment', self.text)[0]
        return self.text