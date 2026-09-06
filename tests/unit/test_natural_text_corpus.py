from pathlib import Path

from puredl.text import (
    build_local_prose_corpus,
    load_text_corpus,
    normalize_natural_text,
    strip_gutenberg_boilerplate,
)


def test_normalize_natural_text_reduces_unicode_punctuation():
    text = "He said \u201chello\u201d\u2014then waited\u2026\n\n\nNext."
    normalized = normalize_natural_text(text)
    assert normalized == 'He said "hello"-then waited...\n\nNext.\n'
    assert "\u2014" not in normalized
    assert "\u2013" not in normalized


def test_strip_gutenberg_boilerplate_keeps_book_body():
    text = (
        "Header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "The story begins here.\n"
        "And continues.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Footer\n"
    )
    stripped = strip_gutenberg_boilerplate(text)
    assert stripped == "The story begins here.\nAnd continues."


def test_build_local_prose_corpus_uses_markdown_without_code(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "# Project\n\nThis is ordinary prose.\n\n```python\nprint('hidden')\n```\n",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("## Note\n\nAnother sentence for the corpus.\n", encoding="utf-8")

    corpus = build_local_prose_corpus(tmp_path)
    assert "ordinary prose" in corpus
    assert "Another sentence" in corpus
    assert "print('hidden')" not in corpus
    assert "# Project" not in corpus


def test_load_text_corpus_custom_file_is_normalized(tmp_path: Path):
    corpus_file = tmp_path / "story.txt"
    corpus_file.write_bytes("A \u201cnormal\u201d sentence.\r\nAnother line.".encode("utf-8"))
    loaded = load_text_corpus(tmp_path, corpus_path=corpus_file)
    assert loaded == 'A "normal" sentence.\nAnother line.\n'
