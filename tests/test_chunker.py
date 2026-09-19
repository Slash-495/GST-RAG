"""
Test LegalSectionChunker with realistic GST text patterns.
"""
import sys
from pathlib import Path
import pymupdf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest import LegalSectionChunker, extract_pages_from_pdf

def test_chunker_with_sample_pdf():
    pdf_path = Path("test_gst_sample.pdf")
    doc = pymupdf.open()

    page1 = doc.new_page()
    text_page1 = """THE CENTRAL GOODS AND SERVICES TAX ACT, 2017

CHAPTER I
PRELIMINARY

1. Short title, extent and commencement.-
(1) This Act may be called the Central Goods and Services Tax Act, 2017.
(2) It extends to the whole of India.
(3) It shall come into force on such date as the Central Government may notify.

2. Definitions.-
In this Act, unless the context otherwise requires,-
(1) "actionable claim" shall have the same meaning as assigned to it in section 130 of the Transfer of Property Act, 1882;
(2) "address of delivery" means the address of the recipient of goods or services.
"""
    page1.insert_text((50, 50), text_page1)

    page2 = doc.new_page()
    text_page2 = """CHAPTER III
LEVY AND COLLECTION OF TAX

7. Scope of supply.-
(1) For the purposes of this Act, the expression "supply" includes-
(a) all forms of supply of goods or services or both such as sale, transfer, barter, exchange, licence, rental, lease or disposal made or agreed to be made for a consideration by a person in the course or furtherance of business;
(b) import of services for a consideration whether or not in the course or furtherance of business;

9. Levy and Collection.-
(1) Subject to the provisions of sub-section (2), there shall be levied a tax called the central goods and services tax on all intra-State supplies of goods or services.

SCHEDULE I
ACTIVITIES TO BE TREATED AS SUPPLY EVEN IF MADE WITHOUT CONSIDERATION
1. Permanent transfer or disposal of business assets where input tax credit has been availed on such assets.
"""
    page2.insert_text((50, 50), text_page2)

    doc.save(str(pdf_path))
    doc.close()

    try:
        pages = extract_pages_from_pdf(pdf_path)
        assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"

        chunker = LegalSectionChunker()
        chunks = chunker.parse_document(pages, source_filename=pdf_path.name)

        sections_found = [c['section'] for c in chunks]
        assert "Section 1" in sections_found, f"Section 1 missing: {sections_found}"
        assert "Section 2" in sections_found, f"Section 2 missing: {sections_found}"
        assert "Section 7" in sections_found, f"Section 7 missing: {sections_found}"
        assert "Section 9" in sections_found, f"Section 9 missing: {sections_found}"
        assert any("SCHEDULE" in s for s in sections_found), f"Schedule missing: {sections_found}"

        print("Chunker verification completed successfully!")
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_continuous_statutory_list_section_17_5():
    """
    Test that continuous legal lists (such as Section 17(5) blocked credits)
    split cleanly along statutory clause boundaries ['\nCHAPTER', '\nSection', '\n(', '\n', '. ', ' ']
    with 15-20% boundary overlap without orphaning clauses.
    """
    section_17_text = """17. Apportionment of credit and blocked credits.-
(1) Where the goods or services or both are used by the registered person partly for the purpose of any business and partly for other purposes, the amount of credit shall be restricted to so much of the input tax as is attributable to the purposes of his business.
(2) Where the goods or services or both are used by the registered person partly for effecting taxable supplies including zero-rated supplies and partly for effecting exempt supplies, the amount of credit shall be restricted.
(5) Notwithstanding anything contained in sub-section (1) of section 16 and subsection (1) of section 18, input tax credit shall not be available in respect of the following, namely:-
(a) motor vehicles for transportation of persons having approved seating capacity of not more than thirteen persons (including the driver), except when they are used for making taxable supplies;
(aa) vessels and aircraft except when they are used for making taxable supplies of such vessels or aircraft;
(ab) services of general insurance, servicing, repair and maintenance in so far as they relate to motor vehicles, vessels or aircraft referred to in clause (a) or clause (aa);
(b) the following supply of goods or services or both-
(i) food and beverages, outdoor catering, beauty treatment, health services, cosmetic and plastic surgery;
(ii) membership of a club, health and fitness centre;
(iii) travel benefits extended to employees on vacation such as leave or home travel concession;
(c) works contract services when supplied for construction of an immovable property (other than plant and machinery);
(d) goods or services or both received by a taxable person for construction of an immovable property on his own account;
(e) goods or services or both on which tax has been paid under section 10;
(f) goods or services or both received by a non-resident taxable person, except on goods imported by him;
(g) goods or services or both used for personal consumption;
(h) goods lost, stolen, destroyed, written off or disposed of by way of gift or free samples; and
(i) any tax paid in accordance with the provisions of sections 74, 129 and 130.
"""
    # Use max_chunk_chars=500 to force subdivision into multiple chunks
    chunker = LegalSectionChunker(max_chunk_chars=500)
    assert 75 <= chunker.chunk_overlap_chars <= 100, f"Overlap should be 15-20%: got {chunker.chunk_overlap_chars}"

    sections = [{
        "source": "CGST_Act_2017.pdf",
        "chapter": "CHAPTER V: INPUT TAX CREDIT",
        "section": "Section 17",
        "title": "Apportionment of credit and blocked credits",
        "pages": [41, 42, 43],
        "text": section_17_text
    }]

    chunks = chunker._subdivide_large_sections(sections)
    print(f"\nSubdivided Section 17 into {len(chunks)} sub-chunks.")
    assert len(chunks) > 1, "Should have subdivided into multiple sub-chunks"

    for i, c in enumerate(chunks):
        content = c["content"]
        raw = c["raw_text"]
        print(f"  Chunk {i+1} length: {len(raw)} chars")
        # Every chunk must preserve the statutory section header
        assert "[CGST_Act_2017.pdf | CHAPTER V: INPUT TAX CREDIT | Section 17: Apportionment of credit and blocked credits]" in content
        assert c["is_subchunk"] is True
        assert c["part"] == i + 1

    # Verify that clauses start cleanly with '(' and are not split mid-word
    for c in chunks[1:]:
        first_line = c["raw_text"].strip().splitlines()[0]
        # Should start with a legal clause boundary e.g. (a), (b), (i), (c)
        assert first_line.startswith("(") or first_line.startswith("17") or any(first_line.startswith(x) for x in ["(1)", "(2)", "(5)", "(a)", "(b)", "(c)", "(d)", "(e)"]), f"Clause boundary failed: {first_line}"

    print("Section 17(5) continuous statutory list chunking test passed successfully!")


if __name__ == "__main__":
    test_chunker_with_sample_pdf()
    test_continuous_statutory_list_section_17_5()
