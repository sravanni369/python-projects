# python-projects

Standalone Python projects. Each file runs on its own with no third-party libraries: `python <file>.py`.

| File | What it does | Concepts |
|------|--------------|----------|
| sentiment_analyser.py | Scores text positive / negative / neutral and reports which words drove the result | Lexicon lookup, intensifier multipliers, negation window, neutral dead zone |
| contact_book.py | Create, search, update, and delete contacts in SQLite | Parameterised SQL, UNIQUE constraint, `sqlite3.Row`, transactions, `rowcount` |

## Notes

**sentiment_analyser.py** — a negation within the preceding three words flips a word's score and dampens it to 0.75, so "not good" reads negative without being treated as strongly as "terrible". Lexicon methods do not handle double negatives: "I don't hate it" scores wrong.

**contact_book.py** — the UNIQUE constraint on `phone` means the database rejects duplicates rather than application code checking first. Defaults to an in-memory database; pass a path to `connect()` to persist.
