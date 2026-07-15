"""Tests for contact_book."""

import pytest

from contact_book import (
    add,
    all_contacts,
    connect,
    count_by_tag,
    delete,
    search,
    update_phone,
)


@pytest.fixture
def con():
    """A fresh in-memory database per test."""
    connection = connect()
    yield connection
    connection.close()


@pytest.fixture
def populated(con):
    add(con, "Asha Rao", "555-0101", "asha@example.com", "work")
    add(con, "Brian Cole", "555-0102", "brian@example.com", "personal")
    add(con, "Chitra Nair", "555-0103", "chitra@example.com", "work")
    return con


def test_add_returns_new_id(con):
    assert add(con, "Asha Rao", "555-0101") == 1


def test_duplicate_phone_is_rejected(populated):
    assert add(populated, "Someone Else", "555-0101") is None


def test_duplicate_phone_does_not_insert_a_row(populated):
    add(populated, "Someone Else", "555-0101")
    assert len(all_contacts(populated)) == 3


def test_whitespace_is_stripped_on_insert(con):
    add(con, "  Asha Rao  ", "  555-0101  ")
    row = all_contacts(con)[0]
    assert row["name"] == "Asha Rao"
    assert row["phone"] == "555-0101"


def test_stripping_makes_padded_duplicates_collide(con):
    add(con, "Asha Rao", "555-0101")
    assert add(con, "Asha Again", " 555-0101 ") is None


def test_tag_defaults_to_personal(con):
    add(con, "Asha Rao", "555-0101")
    assert all_contacts(con)[0]["tag"] == "personal"


def test_search_matches_name(populated):
    assert [c["name"] for c in search(populated, "Asha")] == ["Asha Rao"]


def test_search_is_case_insensitive(populated):
    assert [c["name"] for c in search(populated, "asha")] == ["Asha Rao"]


def test_search_matches_partial_email(populated):
    assert len(search(populated, "example.com")) == 3


def test_search_matches_phone(populated):
    assert [c["name"] for c in search(populated, "0102")] == ["Brian Cole"]


def test_search_returns_empty_for_no_match(populated):
    assert search(populated, "zzzz") == []


def test_search_orders_by_name(populated):
    assert [c["name"] for c in search(populated, "example.com")] == [
        "Asha Rao", "Brian Cole", "Chitra Nair"
    ]


def test_update_phone_reports_one_row(populated):
    assert update_phone(populated, 1, "555-0999") == 1


def test_update_phone_actually_changes_the_value(populated):
    update_phone(populated, 1, "555-0999")
    assert search(populated, "Asha")[0]["phone"] == "555-0999"


def test_update_missing_id_reports_zero_rows(populated):
    assert update_phone(populated, 999, "555-0999") == 0


def test_delete_reports_one_row(populated):
    assert delete(populated, 2) == 1


def test_delete_actually_removes_the_row(populated):
    delete(populated, 2)
    assert [c["name"] for c in all_contacts(populated)] == ["Asha Rao", "Chitra Nair"]


def test_delete_missing_id_reports_zero_rows(populated):
    assert delete(populated, 999) == 0


def test_count_by_tag_groups(populated):
    assert count_by_tag(populated) == {"work": 2, "personal": 1}


def test_count_by_tag_drops_emptied_groups(populated):
    delete(populated, 2)  # the only personal contact
    assert count_by_tag(populated) == {"work": 2}


def test_count_by_tag_on_empty_table(con):
    assert count_by_tag(con) == {}
