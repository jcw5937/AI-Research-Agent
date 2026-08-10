from docx import Document

import writer


def _paragraph_texts(path):
    return [p.text for p in Document(path).paragraphs]


class TestBuildLayerDocumentRendering:
    def test_renders_full_metadata_and_sections(self, tmp_path):
        papers_notes = [{
            "title": "A Real Paper",
            "url": "https://arxiv.org/abs/1234.5678",
            "authors": "Jane Doe, John Smith",
            "venue": "NeurIPS",
            "verified": True,
            "peer_reviewed": True,
            "sections": {"Key Claims": "It works."},
        }]

        path = writer.build_layer_document(
            "Test Layer", papers_notes, str(tmp_path), note_sections=["Key Claims", "Tools"]
        )
        texts = _paragraph_texts(path)

        assert "Test Layer" in texts
        assert "A Real Paper" in texts
        assert "Jane Doe, John Smith — NeurIPS | verified | peer-reviewed" in texts
        assert "https://arxiv.org/abs/1234.5678" in texts
        assert "It works." in texts
        assert "No information extracted." in texts  # Tools section has no content

    def test_missing_authors_and_venue_fall_back(self, tmp_path):
        papers_notes = [{
            "title": "Web Result",
            "url": "",
            "authors": "",
            "venue": "",
            "verified": False,
            "peer_reviewed": False,
            "sections": {},
        }]

        path = writer.build_layer_document("L", papers_notes, str(tmp_path), note_sections=["Key Claims"])
        texts = _paragraph_texts(path)

        assert "Unknown authors — Unknown venue | unverified | not peer-reviewed" in texts

    def test_no_url_means_no_url_paragraph(self, tmp_path):
        papers_notes = [{
            "title": "No URL Paper",
            "url": "",
            "authors": "A",
            "venue": "V",
            "sections": {},
        }]

        path = writer.build_layer_document("L", papers_notes, str(tmp_path), note_sections=["Key Claims"])
        texts = _paragraph_texts(path)

        assert not any(t.startswith("http") for t in texts)

    def test_missing_section_defaults_to_placeholder(self, tmp_path):
        papers_notes = [{"title": "T", "sections": {"Key Claims": "   "}}]

        path = writer.build_layer_document("L", papers_notes, str(tmp_path), note_sections=["Key Claims"])
        texts = _paragraph_texts(path)

        assert "No information extracted." in texts


class TestBuildLayerDocumentNoteSections:
    def test_none_note_sections_falls_back_to_default(self, tmp_path):
        papers_notes = [{"title": "T", "sections": {}}]
        path = writer.build_layer_document("L", papers_notes, str(tmp_path), note_sections=None)
        texts = _paragraph_texts(path)
        for section in writer.DEFAULT_NOTE_SECTIONS:
            assert section in texts

    def test_custom_note_sections_used_instead_of_default(self, tmp_path):
        papers_notes = [{"title": "T", "sections": {}}]
        path = writer.build_layer_document("L", papers_notes, str(tmp_path), note_sections=["Only This"])
        texts = _paragraph_texts(path)
        assert "Only This" in texts
        assert "Key Claims" not in texts


class TestBuildLayerDocumentFilenameSanitization:
    def test_special_characters_are_sanitized(self, tmp_path):
        path = writer.build_layer_document(
            "Weird/Layer:Name*?", [], str(tmp_path), note_sections=["Key Claims"]
        )
        assert path == str(tmp_path / "Weird_Layer_Name__.docx")

    def test_spaces_become_underscores(self, tmp_path):
        path = writer.build_layer_document("My Layer Name", [], str(tmp_path), note_sections=["Key Claims"])
        assert path == str(tmp_path / "My_Layer_Name.docx")

    def test_disallowed_characters_are_replaced_not_stripped(self, tmp_path):
        path = writer.build_layer_document(
            "Weird?Name!", [], str(tmp_path), note_sections=["Key Claims"]
        )
        assert path == str(tmp_path / "Weird_Name_.docx")

    def test_empty_layer_name_falls_back_to_layer(self, tmp_path):
        path = writer.build_layer_document("   ", [], str(tmp_path), note_sections=["Key Claims"])
        assert path == str(tmp_path / "layer.docx")

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        path = writer.build_layer_document("L", [], str(nested), note_sections=["Key Claims"])
        assert nested.is_dir()
        assert path == str(nested / "L.docx")
