"""Developer configuration for the example Streamlit chat app."""

import json
from openai_sdk_helpers.agent.web_search import WebAgentSearch
from openai_sdk_helpers.config import OpenAISettings
from openai_sdk_helpers.response.base import BaseResponse
from openai_sdk_helpers.structure.web_search import WebSearchStructure
from openai_sdk_helpers.structure.prompt import PromptStructure
from openai_sdk_helpers.utils.core import customJSONEncoder


class StreamlitWebSearch(BaseResponse[WebSearchStructure]):
    """Response tuned for a generic chat experience with structured output.

    Methods
    -------
    __init__()
        Configure a general-purpose response session using OpenAI settings.
    """

    def __init__(self) -> None:
        settings = OpenAISettings.from_env()
        super().__init__(
            instructions="Perform web searches and generate reports.",
            tools=[
                PromptStructure.response_tool_definition(
                    tool_name="perform_search",
                    tool_description="Tool to perform web searches and generate reports.",
                )
            ],
            schema=WebSearchStructure.response_format(),
            output_structure=WebSearchStructure,
            tool_handlers={"perform_search": perform_search},
            client=settings.create_client(),
            model=settings.default_model or "gpt-4o-mini",
        )


async def perform_search(tool) -> str:
    """Perform a web search and return structured results."""
    from openai_sdk_helpers.response.tool_call import parse_tool_arguments

    structured_data = parse_tool_arguments(tool.arguments)
    query = structured_data.get("prompt")
    if not query:
        raise ValueError("No prompt provided for web search.")
    web_result = await WebAgentSearch(default_model="gpt-4o-mini").run_web_agent_async(
        query
    )
    return json.dumps(web_result.to_json(), cls=customJSONEncoder)


APP_CONFIG = {
    "response": StreamlitWebSearch,
    "display_title": "Web Search Assistant",
    "description": "Config-driven chat experience for performing web searches.",
}

if __name__ == "__main__":
    web_search_instance = StreamlitWebSearch()
    import asyncio

    result = asyncio.run(
        web_search_instance.run_async("What are the latest advancements in AI?")
    )
    if result:
        print(web_search_instance.get_last_tool_message())
    else:
        print("No result returned.")
    filepath = f"./data/{web_search_instance.name}.{web_search_instance.uuid}.json"
    web_search_instance.save(filepath)
    web_search_instance.close()
