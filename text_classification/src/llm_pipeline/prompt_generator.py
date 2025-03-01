import textwrap
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import TopicClassification

class PromptGenerator:
    def __init__(self, prompt_data):
        self.parser = PydanticOutputParser(pydantic_object=TopicClassification)
        self.prompt_data = prompt_data
        self.question = self._build_question()

        self.prompt = self._build_prompt_template()
        self.retry = self._build_retry_template()
        self.restrict = self._build_restrict_template()

    def _build_question(self):
        guidelines = self.prompt_data['guidelines']

        focusing_section = "\n".join([
            f"- {guidelines['focusing']['instruction']}",
            f"- Categories(MUST BE ONE OF THESE): {' |  '.join(guidelines['focusing']['categories'])}",
            f"- Fallback Rules: {', '.join(guidelines['focusing']['fallback'])}"
        ])

        keywords_section = "\n".join([
            f"- Extract exactly {guidelines['keywords']['count']} keywords that best represent the main content.",
            *[f"- {inst}" for inst in guidelines['keywords']['instructions']],
            f"- Exclusions: {', '.join(guidelines['keywords']['exclusions'])}"
        ])

        content_length_section = "\n".join([
            f"- {guidelines['content_length']['instruction']}",
            f"- Exclude: {guidelines['content_length']['exclude']}"
        ])

        language_section = "\n".join([
            f"- {guidelines['language']['instruction']}",
            f"- Supported languages: {', '.join(guidelines['language']['options'])}"
        ])

        additional_guidelines_section = "\n".join(
            f"- {guideline}" for guideline in self.prompt_data['additional_guidelines']
        )

        question = textwrap.dedent(f"""
            You are a {self.prompt_data['role']}.
            {self.prompt_data['instruction']}

            1. Focusing:
            {focusing_section}

            2. Keywords:
            {keywords_section}

            3. Content Length:
            {content_length_section}

            4. Language:
            {language_section}

            Additional guidelines:
            {additional_guidelines_section}
        """)

        return question.strip()  # Remove unnecessary leading/trailing spaces

    def _build_prompt_template(self):
        return PromptTemplate(
            template="{question}\n{format_instructions}\nContent:\n{content}",
            input_variables=["question", "content"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def _build_retry_template(self):
        return PromptTemplate(
            input_variables=["question", "content"],
            template=(
                "The previous completion did not match the expected schema."
                "You failed to extract exact focusing from the list of categories"
                "Please return a valid output that conforms exactly to the provided format."
                "{question}\n{format_instructions}\nContent:\n{content}"
            ),
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def _build_restrict_template(self):
        return PromptTemplate(
            input_variables=["question", "content"],
            template=(
                "You failed to extract exact focusing from the list of categories AGAIN!"
                "It is your LAST CHANCE to extract exact category from the content only in provided categories"
                "{question}\n{format_instructions}\nContent:\n{content}"
            ),
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
