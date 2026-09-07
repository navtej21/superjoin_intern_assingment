from ingestion.ingest import extract_pages, PageText
from ingestion.chunker import chunk_document
from pathlib import Path

# --- Test 1: synthetic 3-page input, hand-checkable offsets ---
pages = [
    PageText(page_number=1, text="A" * 2000, char_count=2000, low_text=False),
    PageText(page_number=2, text="B" * 2000, char_count=2000, low_text=False),
    PageText(page_number=3, text="C" * 2000, char_count=2000, low_text=False),
]
chunks = chunk_document("synthdoc", pages)
print("Test 1 - synthetic:")
print("  num chunks:", len(chunks))
for c in chunks:
    print(f"  idx={c.chunk_index} page_start={c.page_start} page_end={c.page_end} "
          f"char_offset={c.char_offset} len={len(c.text)} tokens={c.token_count}")

assert chunks[0].page_start == 1 and chunks[0].char_offset == 0
print("  PASS: chunk 0 starts at page 1, offset 0")

# --- Test 2: real PDF, exhaustive exact-offset verification ---
PDF_PATH = Path(r"C:\Users\Navtej\Desktop\starter-datasets\india-macroeconomy\03-imf-india-2025-article-iv-excerpt.pdf")  # <-- point this at one of the starter PDFs

real_pages = extract_pages(PDF_PATH)
real_chunks = chunk_document("realdoc", real_pages)
full_text = "".join(p.text for p in real_pages)

page_start_offset = {}
cursor = 0
for p in real_pages:
    page_start_offset[p.page_number] = cursor
    cursor += len(p.text)

mismatches = 0
empty = 0
for c in real_chunks:
    abs_offset = page_start_offset[c.page_start] + c.char_offset
    expected = full_text[abs_offset: abs_offset + len(c.text)]
    if expected != c.text:
        mismatches += 1
    if not c.text.strip():
        empty += 1

print()
print("Test 2 - real PDF:", PDF_PATH.name)
print("  pages:", len(real_pages))
print("  chunks:", len(real_chunks))
print("  exact-offset mismatches:", mismatches, "(must be 0)")
print("  empty chunks:", empty, "(must be 0)")

assert mismatches == 0, "OFFSET BUG: char_offset does not point to the chunk's real text"
assert empty == 0, "EMPTY CHUNK BUG"
print()
print("ALL TESTS PASSED")