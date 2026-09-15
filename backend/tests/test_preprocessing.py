"""Unit tests for text preprocessing and the tokenizer."""

from app.utils.text_processing import (
    Tokenizer,
    build_sliding_windows,
    normalize_text,
    tokenize,
)


def test_normalize_lowercases_and_strips_punctuation() -> None:
    out = normalize_text("Hello, World! It's a TEST.")
    assert out == "hello , world ! it is a test ."


def test_tokenize_uses_whitespace() -> None:
    assert tokenize("hello , world !") == ["hello", ",", "world", "!"]


def test_sliding_windows_align_inputs_with_targets() -> None:
    ids = [10, 11, 12, 13, 14, 15, 16]
    inputs, targets = build_sliding_windows(ids, sequence_length=3)
    assert len(inputs) == len(targets) == len(ids) - 3
    assert inputs[0] == [10, 11, 12]
    assert targets[0] == 13
    assert inputs[-1] == [13, 14, 15]
    assert targets[-1] == 16


def test_tokenizer_encodes_unknown_as_unk() -> None:
    tokenizer = Tokenizer()
    tokenizer.build_from_corpus([["hello", "world"]], min_freq=1)
    ids = tokenizer.encode(["hello", "rareword", "world"])
    assert ids[0] == tokenizer.word2idx["hello"]
    assert ids[1] == tokenizer.unk_idx
    assert ids[2] == tokenizer.word2idx["world"]


def test_tokenizer_roundtrip(tmp_path) -> None:
    tokenizer = Tokenizer()
    tokenizer.build_from_corpus([["a", "b", "c"], ["b", "c", "d"]], min_freq=1)
    p = tmp_path / "tok.pkl"
    tokenizer.save(str(p))
    loaded = Tokenizer.load(str(p))
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode(["a", "b"]) == tokenizer.encode(["a", "b"])
