import pytest
from extract_data import ExtractData


def test_extract_txt_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world", encoding="utf-8")
    result = ExtractData(str(f)).extract_text()
    assert result == "hello world"


def test_extract_directory(tmp_path):
    (tmp_path / "a.txt").write_text("foo", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bar", encoding="utf-8")
    result = ExtractData(str(tmp_path)).extract_text()
    assert "foo" in result
    assert "bar" in result


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        ExtractData("test.xyz").extract_text()
