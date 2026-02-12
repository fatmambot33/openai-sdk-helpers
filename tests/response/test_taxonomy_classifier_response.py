from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai_sdk_helpers.response.classifier import (
    TaxonomyClassifierResponse,
    classify_taxonomy_response,
)
from openai_sdk_helpers.structure import (
    ClassificationResult,
    ClassificationStopReason,
    ClassificationSummary,
    TaxonomyNode,
)


def _response_with_output(payload: dict[str, object]) -> SimpleNamespace:
    """Return a fake Responses API payload container."""
    return SimpleNamespace(output_text=__import__("json").dumps(payload))


def test_taxonomy_classifier_response_matches_agent_semantics() -> None:
    """Response classifier should mirror classifier-agent traversal outputs."""
    taxonomy = [TaxonomyNode(label="Billing", children=[TaxonomyNode(label="Invoice")])]
    classifier = TaxonomyClassifierResponse(taxonomy=taxonomy, model="gpt-4o-mini")

    fake_client = MagicMock()
    fake_client.responses.create.side_effect = [
        _response_with_output(
            {
                "selected_nodes": ["Billing"],
                "confidence": 0.9,
                "stop_reason": "continue",
                "rationale": "Billing intent identified.",
            }
        ),
        _response_with_output(
            {
                "selected_nodes": ["Billing > Invoice"],
                "confidence": 0.93,
                "stop_reason": "stop",
                "rationale": "Invoice issue is explicit.",
            }
        ),
    ]

    with patch.object(classifier, "_get_client", return_value=fake_client):
        result = classifier.run_sync("I need help with invoice corrections")

    assert isinstance(result, ClassificationResult)
    assert result.stop_reason is ClassificationStopReason.STOP
    assert [node.label for node in result.final_nodes or []] == ["Invoice"]
    assert result.selected_nodes == ["Billing", "Billing > Invoice"]


def test_taxonomy_classifier_response_returns_summary() -> None:
    """Response classifier should return lightweight summary when configured."""
    taxonomy = [TaxonomyNode(label="Support")]
    classifier = TaxonomyClassifierResponse(
        taxonomy=taxonomy,
        model="gpt-4o-mini",
        return_summary=True,
    )

    fake_client = MagicMock()
    fake_client.responses.create.return_value = _response_with_output(
        {
            "selected_nodes": ["Support"],
            "confidence": 0.88,
            "stop_reason": "stop",
            "rationale": "Support issue detected.",
        }
    )

    with patch.object(classifier, "_get_client", return_value=fake_client):
        result = classifier.run_sync("Need customer support")

    assert isinstance(result, ClassificationSummary)
    assert result.full_paths == ["Support"]


def test_taxonomy_classifier_response_supports_enum_taxonomy() -> None:
    """Response classifier should classify against enum-backed taxonomy paths."""

    class TicketTaxonomy(Enum):
        BILLING = "Billing"
        BILLING_INVOICE = "Billing > Invoice"
        SUPPORT = "Support"

    classifier = TaxonomyClassifierResponse(
        taxonomy=TicketTaxonomy,
        model="gpt-4o-mini",
    )

    fake_client = MagicMock()
    fake_client.responses.create.side_effect = [
        _response_with_output(
            {
                "selected_nodes": ["Billing"],
                "confidence": 0.91,
                "stop_reason": "continue",
                "rationale": "Billing match.",
            }
        ),
        _response_with_output(
            {
                "selected_nodes": ["Billing > Invoice"],
                "confidence": 0.95,
                "stop_reason": "stop",
                "rationale": "Invoice match.",
            }
        ),
    ]

    with patch.object(classifier, "_get_client", return_value=fake_client):
        result = classifier.run_sync("Invoice was overcharged")

    assert isinstance(result, ClassificationResult)
    assert [node.label for node in result.final_nodes or []] == ["Invoice"]


def test_classify_taxonomy_response_function_builder() -> None:
    """Top-level helper should run sync classification using the response class."""
    taxonomy = [TaxonomyNode(label="Support")]
    expected = ClassificationResult(stop_reason=ClassificationStopReason.STOP)

    with patch(
        "openai_sdk_helpers.response.classifier.TaxonomyClassifierResponse.run_sync"
    ) as mock_run:
        mock_run.return_value = expected
        result = classify_taxonomy_response(
            content="Need support",
            taxonomy=taxonomy,
            model="gpt-4o-mini",
            max_depth=1,
            confidence_threshold=0.7,
        )

    assert result is expected
    mock_run.assert_called_once_with(
        "Need support",
        max_depth=1,
        confidence_threshold=0.7,
    )
