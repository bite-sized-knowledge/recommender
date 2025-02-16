from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import TopicClassification

class PromptGenerator:
    def __init__(self, prompt_data):
        self.parser = PydanticOutputParser(pydantic_object=TopicClassification)
        self.prompt_data = prompt_data
        self.question = self._build_question()
        self.prompt = self._build_prompt_template()

    def _build_question(self):
        guidelines = self.prompt_data['guidelines']
        question = f"""
        You are a {self.prompt_data['role']}.
        Task: {self.prompt_data['task']}
        {self.prompt_data['instruction']}

        Follow these guidelines strictly:

        1. Focusing:
        - {guidelines['focusing']['instruction']}: {guidelines['focusing']['categories']}
        - {', '.join(guidelines['focusing']['fallback'])}

        2. Keywords:
        - Extract exactly {guidelines['keywords']['count']} keywords that best represent the main content.
        {' '.join([f'- {inst}' for inst in guidelines['keywords']['instructions']])}
        - Exclusions: {', '.join(guidelines['keywords']['exclusions'])}

        3. Content Length:
        - {guidelines['content_length']['instruction']}
        - Exclude: {guidelines['content_length']['exclude']}

        4. Language:
        - {guidelines['language']['instruction']}
        - Supported languages: {', '.join(guidelines['language']['options'])}

        Additional guidelines:
        {' '.join([f'- {guideline}' for guideline in self.prompt_data['additional_guidelines']])}
        """
        return question

    def _build_prompt_template(self):
        return PromptTemplate(
            template="{question}\n{format_instructions}\nContent:\n{content}",
            input_variables=["question", "content"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def generate_query(self, content):
        return self.prompt.format(content=content, question=self.question)