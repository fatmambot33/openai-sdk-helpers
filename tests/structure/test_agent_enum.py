from openai_sdk_helpers.structure import AgentEnum


def test_agent_enum_includes_text_agents():
    """Ensure text agents are represented in the plan enum crosswalk."""

    assert AgentEnum.SUMMARIZER.value == "SummarizerAgent"
    assert AgentEnum.TRANSLATOR.value == "TranslatorAgent"
    assert AgentEnum.VALIDATOR.value == "ValidatorAgent"

    crosswalk = AgentEnum.CROSSWALK()
    assert crosswalk["SUMMARIZER"]["value"] == "SummarizerAgent"
    assert crosswalk["TRANSLATOR"]["value"] == "TranslatorAgent"
    assert crosswalk["VALIDATOR"]["value"] == "ValidatorAgent"
