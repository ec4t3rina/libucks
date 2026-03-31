"""Phase 1 Testing Gate — test_embedding_service.py

Tests output shape, strict L2-normalisation, determinism, and singleton
behaviour for EmbeddingService.

The real all-MiniLM-L6-v2 model is used throughout — no mocking of the
model itself.  The singleton "loads only once" test patches the constructor
at the libucks module level after a forced reset so we can count calls
without triggering a second download.
"""

import numpy as np
import pytest

from libucks.embeddings.embedding_service import EmbeddingService

# The model name used in every test that needs a live model
_MODEL = "all-MiniLM-L6-v2"
_DIM = 384


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Guarantee a clean singleton slate before and after every test."""
    EmbeddingService.reset()
    yield
    EmbeddingService.reset()


@pytest.fixture
def svc() -> EmbeddingService:
    return EmbeddingService.get_instance(_MODEL)


# ---------------------------------------------------------------------------
# embed() — single text
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_output_shape_is_384(self, svc: EmbeddingService):
        vec = svc.embed("hello world")
        assert vec.shape == (_DIM,), f"Expected ({_DIM},), got {vec.shape}"

    def test_output_dtype_is_float32(self, svc: EmbeddingService):
        vec = svc.embed("hello world")
        assert vec.dtype == np.float32

    def test_output_is_l2_normalised(self, svc: EmbeddingService):
        vec = svc.embed("the authentication middleware validates JWTs")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-6, f"Expected L2-norm ≈ 1.0, got {norm}"

    def test_short_text_is_l2_normalised(self, svc: EmbeddingService):
        vec = svc.embed("a")
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6

    def test_long_text_is_l2_normalised(self, svc: EmbeddingService):
        long_text = "word " * 300
        vec = svc.embed(long_text)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6

    def test_deterministic_for_identical_input(self, svc: EmbeddingService):
        text = "database schema migration"
        v1 = svc.embed(text)
        v2 = svc.embed(text)
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_produce_different_vectors(self, svc: EmbeddingService):
        v1 = svc.embed("authentication and JWT tokens")
        v2 = svc.embed("database schema and ORM models")
        assert not np.allclose(v1, v2), "Semantically distinct texts must not embed identically"

    def test_similar_texts_are_closer_than_unrelated(self, svc: EmbeddingService):
        v_auth1 = svc.embed("JWT authentication middleware")
        v_auth2 = svc.embed("token validation and auth flow")
        v_unrel = svc.embed("database schema migration scripts")
        sim_close = float(np.dot(v_auth1, v_auth2))
        sim_far = float(np.dot(v_auth1, v_unrel))
        assert sim_close > sim_far, (
            "Semantically related texts must score higher cosine similarity "
            f"than unrelated ones (got {sim_close:.4f} vs {sim_far:.4f})"
        )


# ---------------------------------------------------------------------------
# embed_batch() — multiple texts
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    def test_output_shape_two_texts(self, svc: EmbeddingService):
        mat = svc.embed_batch(["hello", "world"])
        assert mat.shape == (2, _DIM)

    def test_output_shape_single_text(self, svc: EmbeddingService):
        mat = svc.embed_batch(["only one"])
        assert mat.shape == (1, _DIM)

    def test_output_dtype_is_float32(self, svc: EmbeddingService):
        mat = svc.embed_batch(["a", "b"])
        assert mat.dtype == np.float32

    def test_every_row_is_l2_normalised(self, svc: EmbeddingService):
        texts = [
            "authentication middleware",
            "database schema",
            "HTTP route handlers",
            "frontend components",
        ]
        mat = svc.embed_batch(texts)
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(texts)), atol=1e-6)

    def test_deterministic_for_identical_inputs(self, svc: EmbeddingService):
        texts = ["auth", "db", "api"]
        m1 = svc.embed_batch(texts)
        m2 = svc.embed_batch(texts)
        np.testing.assert_array_equal(m1, m2)

    def test_batch_rows_match_individual_embeds(self, svc: EmbeddingService):
        texts = ["auth middleware", "ORM models"]
        mat = svc.embed_batch(texts)
        for i, text in enumerate(texts):
            np.testing.assert_array_almost_equal(mat[i], svc.embed(text))


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_instance_returns_same_object(self):
        svc1 = EmbeddingService.get_instance(_MODEL)
        svc2 = EmbeddingService.get_instance(_MODEL)
        assert svc1 is svc2

    def test_model_constructor_called_only_once(self, monkeypatch):
        """SentenceTransformer must be constructed exactly once even when
        get_instance() is called three times.

        We patch the *module-level name* in libucks.embeddings.embedding_service
        (not the class itself) so that monkeypatch guarantees a clean restore
        after the test — patching __new__ on a third-party class can leak.
        """
        import libucks.embeddings.embedding_service as emb_module

        call_count = 0
        RealST = emb_module.SentenceTransformer

        def tracking_ST(model_name):
            nonlocal call_count
            call_count += 1
            return RealST(model_name)

        monkeypatch.setattr(emb_module, "SentenceTransformer", tracking_ST)

        svc1 = EmbeddingService.get_instance(_MODEL)
        svc2 = EmbeddingService.get_instance(_MODEL)
        svc3 = EmbeddingService.get_instance(_MODEL)

        assert call_count == 1, (
            f"SentenceTransformer was constructed {call_count} times; expected exactly 1"
        )
        assert svc1 is svc2 is svc3

    def test_reset_allows_fresh_instance(self):
        svc1 = EmbeddingService.get_instance(_MODEL)
        EmbeddingService.reset()
        svc2 = EmbeddingService.get_instance(_MODEL)
        # After reset a new object must have been created
        assert svc1 is not svc2

    def test_instance_is_embedding_service(self):
        svc = EmbeddingService.get_instance(_MODEL)
        assert isinstance(svc, EmbeddingService)

    def test_get_instance_without_args_uses_default_model(self):
        svc = EmbeddingService.get_instance()
        # Default model still produces the right output shape
        assert svc.embed("test").shape == (_DIM,)


# ---------------------------------------------------------------------------
# L2-normalisation contract — adversarial edge cases
# ---------------------------------------------------------------------------

class TestL2NormalisationContract:
    """The cosine similarity in Phase 2 is computed as dot(q, c) which is only
    equal to cosine similarity when both vectors are L2-normalised.  These
    tests lock that contract down for edge cases that trip up naive wrappers."""

    @pytest.mark.parametrize("text", [
        "a",
        "   ",                        # whitespace only
        "x" * 512,                    # very long single token
        "日本語のテキスト",            # non-ASCII
        "def authenticate(token):\n    return validate_jwt(token)",  # code
        "0" * 100,                    # repeated character
    ])
    def test_normalised_for_edge_case_text(self, svc: EmbeddingService, text: str):
        vec = svc.embed(text)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-6, f"Norm was {norm} for input {text!r:.40}"
