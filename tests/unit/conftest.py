"""Unit-test conftest: stub out packages that fail to import in the CI environment.

sentence_transformers triggers a chain:
  sentence_transformers → transformers.integrations.peft
                        → transformers.core_model_loading
                        → transformers.integrations.accelerate
                        → NameError: name 'nn' is not defined (line 62)

All unit tests that touch EmbeddingService already patch get_instance(), so
the real SentenceTransformer implementation is never called. Stubbing the
package here lets the module-level import in embedding_service.py succeed.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _stub_sentence_transformers() -> None:
    if "sentence_transformers" in sys.modules:
        return
    stub = MagicMock()
    stub.SentenceTransformer = MagicMock
    sys.modules["sentence_transformers"] = stub


_stub_sentence_transformers()
