"""Contact book backed by SQLite.

Create, search, update, and delete contacts in a local SQLite file.
Demonstrates parameterised SQL, a UNIQUE constraint, and transactions.

No third-party libraries. Run: python contact_book.py
"""

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    phone   TEXT NOT NULL UNIQUE,
    email   TEXT,
    tag     TEXT DEFAULT 'personal'
)
"""


def connect(path=":memory:"):
    """Open a connection with dict-like rows and the schema applied."""
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    return con


def add(con, name, phone, email=None, tag="personal"):
    """Insert a contact. Returns the new id, or None if the phone exists."""
    try:
        with con:
            cur = con.execute(
                "INSERT INTO contacts (name, phone, email, tag) VALUES (?, ?, ?, ?)",
                (name.strip(), phone.strip(), email, tag),
            )
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def search(con, term):
    """Find contacts whose name, phone, or email contains term."""
    like = f"%{term.strip()}%"
    rows = con.execute(
        """SELECT * FROM contacts
           WHERE name LIKE ? OR phone LIKE ? OR email LIKE ?
           ORDER BY name""",
        (like, like, like),
    ).fetchall()
    return [dict(r) for r in rows]


def update_phone(con, contact_id, new_phone):
    """Change a contact's phone. Returns rows affected."""
    with con:
        cur = con.execute(
            "UPDATE contacts SET phone = ? WHERE id = ?",
            (new_phone.strip(), contact_id),
        )
    return cur.rowcount


def delete(con, contact_id):
    """Remove a contact. Returns rows affected."""
    with con:
        cur = con.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
    return cur.rowcount


def all_contacts(con):
    """Return every contact, name order."""
    return [dict(r) for r in con.execute("SELECT * FROM contacts ORDER BY name")]


def count_by_tag(con):
    """Group contacts by tag."""
    rows = con.execute(
        "SELECT tag, COUNT(*) AS n FROM contacts GROUP BY tag ORDER BY n DESC"
    ).fetchall()
    return {r["tag"]: r["n"] for r in rows}


if __name__ == "__main__":
    con = connect()

    add(con, "Asha Rao", "555-0101", "asha@example.com", "work")
    add(con, "Brian Cole", "555-0102", "brian@example.com", "personal")
    add(con, "Chitra Nair", "555-0103", "chitra@example.com", "work")

    duplicate = add(con, "Asha Rao Again", "555-0101", "dupe@example.com")
    print("duplicate phone rejected:", duplicate is None)

    print("\nall contacts:")
    for c in all_contacts(con):
        print(f"  {c['id']} {c['name']:<12} {c['phone']:<10} {c['tag']}")

    print("\nsearch 'example.com':", len(search(con, "example.com")), "hits")
    print("search 'asha':", [c["name"] for c in search(con, "asha")])

    print("\nupdate phone:", update_phone(con, 1, "555-0999"), "row")
    print("  Asha now:", search(con, "Asha")[0]["phone"])

    print("\ndelete:", delete(con, 2), "row")
    print("count by tag:", count_by_tag(con))

    con.close()
