from langchain_openai import ChatOpenAI
from .prompt_generator import PromptGenerator
from .config_loader import load_config
from .schemas import TopicClassification

class LangChainModel:
    def __init__(self):
        self.prompt_data = load_config("../config/prompt.yaml")
        self.prompt_generator = PromptGenerator(self.prompt_data)
        self.model = ChatOpenAI(model_name=self.prompt_data['model'])

    def predict(self, text: str) -> TopicClassification:
        query = self.prompt_generator.generate_query(text)
        output = self.model.predict(query)
        return self.prompt_generator.parser.parse(output)