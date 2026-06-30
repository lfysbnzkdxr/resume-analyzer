"""Tests for Chinese resume text splitter."""

from src.rag.splitter import split_text, _find_section_boundaries


class TestFindSectionBoundaries:
    def test_empty_text(self):
        assert _find_section_boundaries("") == [0]

    def test_single_section(self):
        b = _find_section_boundaries("工作经验：写了三年代码")
        assert b == [0]

    def test_multiple_sections(self):
        text = "个人信息：张三\n\n技能：Python\n\n工作经验：写了三年代码"
        b = _find_section_boundaries(text)
        assert len(b) == 3
        assert b[0] == 0  # first section (personal info)

    def test_english_headers(self):
        text = "Summary: experienced\nSkills: python\nExperience: 3 years"
        b = _find_section_boundaries(text)
        assert len(b) == 3


class TestSplitText:
    def test_empty(self):
        assert split_text("") == []

    def test_whitespace_only(self):
        assert split_text("   ") == []

    def test_short_text(self):
        chunks = split_text("个人信息：张三", chunk_size=500)
        assert len(chunks) == 1
        assert "张三" in chunks[0]

    def test_preserves_all_content(self):
        text = "个人信息：张三\n\n技能：Python, Java\n\n工作经验：写了三年代码"
        chunks = split_text(text, chunk_size=500)
        combined = "".join(chunks)
        assert "张三" in combined
        assert "Python" in combined
        assert "三年" in combined

    def test_chunk_size_respected(self):
        """Each chunk should not exceed chunk_size significantly."""
        text = (
            "个人信息：张三\n\n"
            + "技能：" + "a" * 300 + "\n\n"
            + "工作经验：" + "b" * 300 + "\n\n"
            + "项目经验：" + "c" * 300
        )
        chunks = split_text(text, chunk_size=200, chunk_overlap=20)
        for c in chunks:
            assert len(c) <= 350  # allow some overlap margin

    def test_no_section_headers(self):
        text = "这是一段没有章节标题的纯文本。" * 20
        chunks = split_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 1

    def test_small_sections_merged(self):
        """Small consecutive sections should be merged into one chunk."""
        text = "姓名：张三\n\n技能：Python\n\nJava"
        chunks = split_text(text, chunk_size=500)
        assert len(chunks) == 1
