from openai_sdk_helpers.agent.classifier import TaxonomyClassifierAgent
from openai_sdk_helpers.structure.classification import (
    ClassificationResult,
    TaxonomyNode,
)
from agents.model_settings import ModelSettings

# Define a hierarchical taxonomy
taxonomy = [
    TaxonomyNode(
        label="Billing",
        children=[
            TaxonomyNode(label="Invoice"),
            TaxonomyNode(label="Payment"),
        ],
    ),
    TaxonomyNode(
        label="Support",
        children=[
            TaxonomyNode(label="Technical Issue"),
            TaxonomyNode(label="Account Issue"),
        ],
    ),
]

agent = TaxonomyClassifierAgent(
    model="gpt-4o", taxonomy=taxonomy, model_settings=ModelSettings(temperature=0)
)

text = "My last payment failed and I need help fixing it."


# Use run_agent (async) for classification
import asyncio


def print_result(result: ClassificationResult, label):
    print(f"\n--- {label} ---")
    print(f"Classification path:{result.to_lightweight_summary()}")
    print(f"Raw result:\n{result}")


async def main():
    # Async run_agent
    result_async = await agent.run_async(text)


    print_result(result_async, "run_agent (async)")


if __name__ == "__main__":
    asyncio.run(main())
