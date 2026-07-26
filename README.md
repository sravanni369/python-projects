# python-projects

Standalone Python projects. Single-file projects run with no third-party libraries: `python <file>.py`. Projects in their own folder have a README and a `requirements.txt`.

| File | What it does | Concepts |
|------|--------------|----------|
| sentiment_analyser.py | Scores text positive / negative / neutral and reports which words drove the result | Lexicon lookup, intensifier multipliers, negation window, neutral dead zone |
| contact_book.py | Create, search, update, and delete contacts in SQLite | Parameterised SQL, UNIQUE constraint, `sqlite3.Row`, transactions, `rowcount` |
| pm25-forecast/ | Forecasts PM2.5 air pollution 24 hours ahead for six cities in India and the USA | PyTorch GRU, chronological train/val/test split, persistence baseline, live Open-Meteo API |

## Notes

**sentiment_analyser.py** — a negation within the preceding three words flips a word's score and dampens it to 0.75, so "not good" reads negative without being treated as strongly as "terrible". Lexicon methods do not handle double negatives: "I don't hate it" scores wrong.

**contact_book.py** — the UNIQUE constraint on `phone` means the database rejects duplicates rather than application code checking first. Defaults to an in-memory database; pass a path to `connect()` to persist.

**pm25-forecast/** — beats a persistence baseline by 3.8% MAE and 8.4% RMSE on held-out hours, but loses to it on MAE in Delhi and Kolkata, the two highest-variance cities. Numbers and limitations are in the project README; the console output they came from is in `run_log.txt`.
