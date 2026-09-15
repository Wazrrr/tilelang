"""Compatibility facade for portable primitive profile validation."""

from tiletune_core.profile_schema import (
    RATE_FIELDS as RATE_FIELDS,
    LATENCY_FIELDS as LATENCY_FIELDS,
    CONSUMER_RATE_FIELDS as CONSUMER_RATE_FIELDS,
    PROFILE_METADATA_FIELDS as PROFILE_METADATA_FIELDS,
    validate_performance_model as validate_performance_model,
)
