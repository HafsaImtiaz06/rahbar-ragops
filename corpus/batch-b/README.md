# Batch B: documents with planted problems

Used for RAGOps parts A, D4 and E3. Every file is synthetic and carries a banner saying so.
Add them to a corpus that `corpus/fetch.py` has already filled, then verify:

```
python corpus/batch_b.py add --corpus corpus/raw
cd indexer && python -m pipeline.verify --corpus ../corpus/raw --no-move   # report only
python corpus/batch_b.py remove --corpus corpus/raw                         # undo
```

`python corpus/batch_b.py list` prints this table. `batch_b.json` holds the same data and
`indexer/tests/test_batch_b.py` asserts that every row behaves as listed.

| File | Planted problem | Caught by | Outcome |
| --- | --- | --- | --- |
| `library-rules-2026.md` | clean new document | none | indexed |
| `campus-shuttle-schedule-2026.md` | clean new document | none | indexed |
| `library-rules-2026-copy.md` | exact duplicate | uniqueness | quarantined |
| `fee-information-prospectus-2026-copy.pdf` | exact duplicate of the real PDF | uniqueness | quarantined |
| `library-rules-2026-reworded.md` | near-duplicate (two words changed) | uniqueness | quarantined |
| `hostel-rules-2026-truncated.md` | ends mid-sentence | completeness | quarantined |
| `hostel-scan-corrupt.pdf` | corrupt PDF | quality | quarantined |
| `notice-board-undated.md` | no title, no date or version | completeness | quarantined |
| `fee-information-prospectus-2027.md` | newer version of Fee Information | recency | indexed; the 2026 PDF is archived |
| `campus-notices-injected.md` | injected instruction | none (guardrail, part D4) | indexed |
| `exam-hall-seating-garbled.txt` | garbled text | quality | quarantined |
| `blank-meeting-notes.md` | heading only, no text | quality | quarantined |

The copy of the real PDF is made from `corpus/raw/fee-information-prospectus-2026.pdf`, so it
is skipped (with a message) when that file is missing.
