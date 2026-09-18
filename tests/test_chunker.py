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

if __name__ == "__main__":
    test_chunker_with_sample_pdf()
