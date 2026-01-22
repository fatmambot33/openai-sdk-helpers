"""Docstring for scratch.textextract."""

import textwrap
from dotenv import load_dotenv
from openai_sdk_helpers.extract.extractor import DocumentExtractor
from openai_sdk_helpers.structure.extraction import (
    AttributeStructure,
    Document,
    ExampleData,
    Extraction,
)

load_dotenv()

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent(
    """Extract characters, emotions, and relationships in order of appearance. Use exact text for extractions. Do not paraphrase or overlap entities.Provide meaningful attributes for each entity to add context."""
)

# 2. Provide a high-quality example to guide the model

examples = [
    ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes=[
                    AttributeStructure(
                        key="emotional_state",
                        value="wonder",
                    )
                ],
            ),
            Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes=[
                    AttributeStructure(
                        key="feeling",
                        value="gentle awe",
                    )
                ],
            ),
            Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes=[
                    AttributeStructure(
                        key="type",
                        value="metaphor",
                    )
                ],
            ),
        ],
    )
]
# The input text to be processed
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"
doc = Document(text=input_text)

# 3. Initialize the DocumentExtractor with the prompt and examples
extractor = DocumentExtractor(
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",
    max_workers=2,
)
# 4. Perform the extraction
annotated_documents = extractor.extract(doc)
# 5. Display the results
for doc in annotated_documents:
    print("Extracted Entities:")
    for extraction in doc:
        print(doc)
