"""opensearch — connection/setup + the fingerprint deduplication index."""

from hippius_validation.opensearch.client import ensure_index, get_client, health
from hippius_validation.opensearch.fingerprints import find_duplicate, index_fingerprint

__all__ = ["get_client", "health", "ensure_index", "find_duplicate", "index_fingerprint"]
