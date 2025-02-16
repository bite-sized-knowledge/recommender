import sys
sys.path.append("../src")
from preprocessing \
    import WoowahanProcessor, TossProcessor, MediumProcessor, KakaoProcessor, OliveYoungProcessor

def postprocess_by_blog_id(text, blog_id):
    processors = {
        1: WoowahanProcessor,
        2: TossProcessor,
        3: MediumProcessor,
        4: KakaoProcessor,
        5: OliveYoungProcessor
    }
    processor_class = processors.get(blog_id)
    if not processor_class:
        raise ValueError(f"Unsupported blog_id: {blog_id}")
    processor = processor_class(text, blog_id)
    return processor.process()