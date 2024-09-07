import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser, # Transform into a JSON object
    PydanticToolsParser # Transform into a Pydantic object
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion

load_dotenv()


llm = ChatOpenAI(model="gpt-4-turbo-preview")
parser_json = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.
Current time: {current_time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format."
        ),
    ]
).partial(   
    current_time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer.",
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="White about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )

    chain = (
        first_responder_prompt_template 
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") 
        | parser_pydantic
    )