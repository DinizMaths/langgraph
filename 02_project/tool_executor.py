import json

from typing import List

from collections import defaultdict

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolInvocation, ToolExecutor

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from schemas import AnswerQuestion, Reflection

from chains import parser_json

from dotenv import load_dotenv


load_dotenv()

search = DuckDuckGoSearchAPIWrapper(max_results=5)
duckduckgo_tool = DuckDuckGoSearchResults(api_wrapper=search)
tool_executor = ToolExecutor([duckduckgo_tool])

def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser_json.invoke(tool_invocation)

    ids = []
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="duckduckgo_results_json",
                    tool_input=query
                )
            )

            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations)

    outputs_map = defaultdict(dict)
    
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    tool_messages = []

    for id_, mapped_output in outputs_map.items():
        tool_messages.append(
            ToolMessage(
                content=json.dumps(mapped_output),
                tool_call_id=id_
            )
        )

    return tool_messages


if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc problem domain,"
        " list startups that do that and raised capital."
    )

    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-Powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_id",
    )

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_id",
                    }
                ]
            )
        ]
    )

    print(raw_res)