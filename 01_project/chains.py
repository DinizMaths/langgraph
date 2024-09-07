from langchain_core.prompts import (
    ChatPromptTemplate, # Hold the content that we either send to LLM as humans or that we receibe back from LLM as an answer
    MessagesPlaceholder # Give us flexibility to put here a placeholder for future messages that we are going to get
)
from langchain_openai import ChatOpenAI # Import the Constructor of the OpenAI Chat

from dotenv import load_dotenv

load_dotenv()

# Define the reflection prompt that will be used to critique the generated text -> how it can be better
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate a critique and recommendations for the user's tweet. "
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the generator prompt that will be used to generate the text -> the tweet
generator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post for the user's request. "
            "If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()
generation_chain = generator_prompt | llm
reflection_chain = reflection_prompt | llm
