import pytest

from backend.ingest.loaders import chunk_text


def test_chunk_text_basic_spacing():
    text = " ".join(str(i) for i in range(1, 11))
    chunks = chunk_text(text, chunk_size=3, chunk_overlap=1)
    # Expect sliding windows of length 3 with step 2
    assert chunks[:3] == ["1 2 3", "3 4 5", "5 6 7"]
    # Last chunk should include remaining tokens and not be empty
    assert chunks[-1] == "9 10"
    # All chunks should have no leading/trailing spaces
    assert all(c == c.strip() for c in chunks)


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap, expected_error",
    [
        (0, 0, "chunk_size must be greater than 0"),
        (5, -1, "chunk_overlap must be non-negative"),
        (5, 5, "chunk_overlap must be smaller than chunk_size"),
        (3, 4, "chunk_overlap must be smaller than chunk_size"),
    ],
)
def test_chunk_text_invalid_params(chunk_size, chunk_overlap, expected_error):
    text = "sample text"
    with pytest.raises(ValueError) as exc:
        chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert expected_error in str(exc.value)
