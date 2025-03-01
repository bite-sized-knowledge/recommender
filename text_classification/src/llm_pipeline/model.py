from langchain_openai import ChatOpenAI
from .prompt_generator import PromptGenerator
from langchain.output_parsers import RetryOutputParser
from langchain.prompts import PromptTemplate
from .config_loader import load_config
from .schemas import TopicClassification

class LangChainModel:
    def __init__(self):
        self.prompt_data = load_config("../config/prompt.yaml")
        self.prompt_generator = PromptGenerator(self.prompt_data)
        self.model = ChatOpenAI(
            model_name=self.prompt_data['model'],
            temperature=1
        )
        self.back_up = ChatOpenAI(
            model_name=self.prompt_data['model'],
            temperature=0.5
        )

        self.restrict_back_up = ChatOpenAI(
            model_name=self.prompt_data['model'],
            temperature=0.0
        )


    def predict(self, text: str) -> TopicClassification:
        primary_chain = self.prompt_generator.prompt | self.model | self.prompt_generator.parser
        fallback_chain = self.prompt_generator.retry | self.back_up | self.prompt_generator.parser
        restrict_chain = self.prompt_generator.restrict | self.restrict_back_up | self.prompt_generator.parser

        chain = primary_chain.with_fallbacks([fallback_chain, restrict_chain])
        
        return chain.invoke({
            "question": self.prompt_generator.question,
            "content": text
        })
