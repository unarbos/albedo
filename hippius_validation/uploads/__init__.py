"""uploads — publish validation artifacts (fingerprint / duplicate JSON) to Hippius S3."""

from hippius_validation.uploads.artifacts import put_fault, update_fingerprint_corpus

__all__ = ["update_fingerprint_corpus", "put_fault"]
