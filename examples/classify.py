from openai_sdk_helpers.agent.classifier import TaxonomyClassifierAgent
from openai_sdk_helpers.structure.classification import (
    ClassificationResult,
    ClassificationSummary,
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


async def main():
    # Async run_agent
    result_async = await agent.run_async(text)
    print(result_async)


if __name__ == "__main__":
    asyncio.run(main())
