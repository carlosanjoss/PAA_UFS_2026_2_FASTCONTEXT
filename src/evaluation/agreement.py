"""Inter-annotator agreement metrics for FastContext."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgreementMetrics:
    """Store binary inter-annotator agreement statistics."""

    count: int
    agreements: int
    disagreements: int
    agreement_rate: float
    cohen_kappa: float | None


def calculate_agreement(
    labels_a: Sequence[int],
    labels_b: Sequence[int],
) -> AgreementMetrics:
    """Calculate agreement rate and Cohen's kappa for binary labels.

    Cohen's kappa is undefined when the expected agreement is exactly 1.0,
    which occurs for degenerate marginals such as two annotators assigning
    the same single class to every item. In that case, ``cohen_kappa`` is
    returned as ``None`` while the observed agreement rate remains valid.
    """
    normalized_a = _validate_labels(
        labels_a,
        name="labels_a",
    )

    normalized_b = _validate_labels(
        labels_b,
        name="labels_b",
    )

    if (
        len(normalized_a)
        != len(normalized_b)
    ):
        raise ValueError(
            "Annotation label sequences "
            "must have the same length."
        )

    if not normalized_a:
        raise ValueError(
            "At least one paired annotation "
            "is required."
        )

    count = len(
        normalized_a
    )

    agreements = sum(
        label_a == label_b
        for label_a, label_b in zip(
            normalized_a,
            normalized_b,
            strict=True,
        )
    )

    disagreements = (
        count
        - agreements
    )

    observed_agreement = (
        agreements
        / count
    )

    positive_a = sum(
        normalized_a
    )

    positive_b = sum(
        normalized_b
    )

    negative_a = (
        count
        - positive_a
    )

    negative_b = (
        count
        - positive_b
    )

    expected_agreement = (
        (
            positive_a
            / count
        )
        * (
            positive_b
            / count
        )
        + (
            negative_a
            / count
        )
        * (
            negative_b
            / count
        )
    )

    kappa: float | None

    if expected_agreement == 1.0:
        kappa = None
    else:
        kappa = (
            observed_agreement
            - expected_agreement
        ) / (
            1.0
            - expected_agreement
        )

    return AgreementMetrics(
        count=count,
        agreements=agreements,
        disagreements=disagreements,
        agreement_rate=(
            observed_agreement
        ),
        cohen_kappa=kappa,
    )


def _validate_labels(
    labels: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    """Validate a binary annotation sequence."""
    normalized: list[int] = []

    for label in labels:
        if isinstance(
            label,
            bool,
        ) or not isinstance(
            label,
            int,
        ):
            raise TypeError(
                f"{name} must contain integers."
            )

        if label not in {
            0,
            1,
        }:
            raise ValueError(
                f"{name} must contain only 0 or 1."
            )

        normalized.append(
            label
        )

    return tuple(
        normalized
    )
