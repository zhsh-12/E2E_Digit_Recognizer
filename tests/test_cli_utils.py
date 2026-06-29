"""Tests for cli/utils.py — CLI helper functions (pure functions, no I/O)."""
import pytest
from pathlib import Path
from cli.utils import extract_true_label, detect_label_mode, get_image_files


class TestExtractTrueLabel:
    """extract_true_label parses [N] from filenames."""

    def test_basic_pattern(self):
        assert extract_true_label("img_40[2].png") == 2

    def test_multi_digit_label(self):
        assert extract_true_label("test[12].jpg") == 12

    def test_label_at_start(self):
        assert extract_true_label("[0]_hello.png") == 0

    def test_no_pattern_raises(self):
        with pytest.raises(ValueError, match="No label pattern"):
            extract_true_label("no_label.png")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            extract_true_label("")


class TestDetectLabelMode:
    """detect_label_mode checks first 5 files for [N] patterns."""

    def test_all_labeled(self):
        files = [
            Path("img_1[0].png"),
            Path("img_2[1].jpg"),
            Path("img_3[2].bmp"),
        ]
        assert detect_label_mode(files) is True

    def test_no_labels(self):
        files = [
            Path("img_1.png"),
            Path("img_2.jpg"),
            Path("img_3.bmp"),
        ]
        assert detect_label_mode(files) is False

    def test_mixed_first_has_label(self):
        files = [
            Path("img[9].png"),
            Path("no_label.jpg"),
            Path("another_no_label.bmp"),
        ]
        assert detect_label_mode(files) is True

    def test_empty_list(self):
        assert detect_label_mode([]) is False


class TestGetImageFiles:
    """get_image_files filters directory listing by image_extensions."""

    def test_filters_correct_extensions(self, tmp_path):
        # Create test files
        (tmp_path / "a.png").touch()
        (tmp_path / "b.jpg").touch()
        (tmp_path / "c.jpeg").touch()
        (tmp_path / "d.txt").touch()
        (tmp_path / "e.csv").touch()

        result = get_image_files(tmp_path)
        names = {p.name for p in result}
        assert names == {"a.png", "b.jpg", "c.jpeg"}

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "test.JPG").touch()
        (tmp_path / "test.PNG").touch()

        result = get_image_files(tmp_path)
        assert len(result) == 2

    def test_empty_directory(self, tmp_path):
        result = get_image_files(tmp_path)
        assert result == []

    def test_no_image_files(self, tmp_path):
        (tmp_path / "readme.md").touch()
        (tmp_path / "data.csv").touch()

        result = get_image_files(tmp_path)
        assert result == []
