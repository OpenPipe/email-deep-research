import unittest
import os
from pathlib import Path

from email_search_tools import EmailSearchTools, SearchResult
from types_enron import Email
from local_email_db import DEFAULT_DB_PATH, generate_database

# Test data based on exploration of the database
TEST_INBOX = "phillip.allen@enron.com"
TEST_SENDER = "mike.grigsby@enron.com"
TEST_RECIPIENT = "ina.rangel@enron.com"


def setup_module():
    """Ensure database exists before running tests."""
    db_path = Path(DEFAULT_DB_PATH)
    if not db_path.exists():
        print(f"Database not found at {db_path}. Generating database...")
        generate_database()
    assert db_path.exists(), f"Database file {db_path} should exist"


def test_search_emails_with_gas_keyword():
    """Test searching emails with 'gas' keyword."""
    email_tools = EmailSearchTools()
    results = email_tools.search_emails(
        inbox=TEST_INBOX, keywords=["gas"], max_results=5
    )
    assert len(results) > 0, "Should find emails with keyword 'gas'"
    for result in results:
        assert "gas" in result.snippet.lower()


def test_search_emails_with_forecast_keyword():
    """Test searching emails with 'forecast' keyword."""
    email_tools = EmailSearchTools()
    results = email_tools.search_emails(
        inbox=TEST_INBOX, keywords=["forecast"], max_results=5
    )
    assert len(results) > 0, "Should find emails with keyword 'forecast'"
    for result in results:
        assert "forecast" in result.snippet.lower()


def test_search_emails_with_from_filter():
    """Test searching emails with from_addr filter."""
    email_tools = EmailSearchTools()
    # Use a query that we can easily check the sender
    results = email_tools.search_emails(
        inbox=TEST_INBOX, keywords=["meeting"], from_addr=TEST_SENDER, max_results=5
    )

    assert len(results) > 0, "Should find emails with from_addr filter"

    # Verify each result actually matches the from_addr filter
    for result in results:
        # Get the full email to check the from_address
        email = email_tools.read_email(result.message_id)
        assert email is not None, f"Could not retrieve email {result.message_id}"
        assert (
            email.from_address == TEST_SENDER
        ), f"Email from_address should be {TEST_SENDER}, got {email.from_address}"


def test_search_emails_with_to_filter():
    """Test searching emails with to_addr filter."""
    email_tools = EmailSearchTools()
    results = email_tools.search_emails(
        inbox=TEST_INBOX, keywords=["meeting"], to_addr=TEST_RECIPIENT, max_results=5
    )

    assert len(results) > 0, "Should find emails with to_addr filter"

    # Verify each result actually matches the to_addr filter
    for result in results:
        # Get the full email to check the recipient addresses
        email = email_tools.read_email(result.message_id)
        assert email is not None, f"Could not retrieve email {result.message_id}"
        # Check if TEST_RECIPIENT is in any of the recipient lists
        recipient_addresses = (
            email.to_addresses + email.cc_addresses + email.bcc_addresses
        )
        assert (
            TEST_RECIPIENT in recipient_addresses
        ), f"Email should have {TEST_RECIPIENT} as recipient"


def test_search_emails_with_date_filters():
    """Test searching emails with date range filters."""
    email_tools = EmailSearchTools()
    # Let's use a date range where we know emails exist
    sent_after = "2000-08-01"
    sent_before = "2000-09-01"

    results = email_tools.search_emails(
        inbox=TEST_INBOX,
        keywords=["meeting"],
        sent_after=sent_after,
        sent_before=sent_before,
        max_results=5,
    )

    assert len(results) > 0, "Should find emails with date filters"

    # Verify each result actually falls within the date range
    for result in results:
        # Get the full email to check the date
        email = email_tools.read_email(result.message_id)
        assert email is not None, f"Could not retrieve email {result.message_id}"

        # Parse the date from the email
        email_date = email.date.split()[0]  # Get just the date part (YYYY-MM-DD)

        # Check if the date is within the specified range
        assert (
            email_date >= sent_after
        ), f"Email date {email_date} should be >= {sent_after}"
        assert (
            email_date < sent_before
        ), f"Email date {email_date} should be < {sent_before}"


def test_search_emails_nonexistent_keyword():
    """Test searching with keyword that doesn't exist."""
    email_tools = EmailSearchTools()
    results = email_tools.search_emails(
        inbox=TEST_INBOX, keywords=["nonexistentxyzkeywordasdf"], max_results=5
    )
    assert len(results) == 0, "Should not find emails with nonexistent keyword"


def test_search_emails_no_keywords():
    """Test searching with empty keywords list."""
    email_tools = EmailSearchTools()
    results = email_tools.search_emails(inbox=TEST_INBOX, keywords=[], max_results=5)
    assert len(results) == 0, "Should return empty list when no keywords provided"


def test_read_email_success():
    """Test retrieving a single email by its ID."""
    email_tools = EmailSearchTools()

    # Use a hardcoded ID we know exists from our exploration
    test_email_id = 15  # This is a known ID from our investigation

    email = email_tools.read_email(str(test_email_id))

    assert email is not None, "Should retrieve email successfully"
    assert email.message_id == str(test_email_id), "Message ID should match"
    assert hasattr(email, "from_address"), "Email should have from_address attribute"
    assert hasattr(email, "subject"), "Email should have subject attribute"
    assert hasattr(email, "body"), "Email should have body attribute"
    assert hasattr(email, "to_addresses"), "Email should have to_addresses attribute"


def test_read_email_nonexistent():
    """Test retrieving an email that doesn't exist."""
    email_tools = EmailSearchTools()
    email = email_tools.read_email("nonexistent_id_12345")
    assert email is None, "Should return None for nonexistent email ID"


def test_database_connection_failures():
    """Test connecting to a non-existent database file."""
    # Point to a non-existent db file
    email_tools = EmailSearchTools(db_path="/nonexistent/path/db.sqlite")

    # Both operations should return empty/None results
    search_results = email_tools.search_emails(
        inbox=TEST_INBOX, keywords=["test"], max_results=5
    )
    assert (
        len(search_results) == 0
    ), "Should return empty list when database doesn't exist"

    email = email_tools.read_email("some_id")
    assert email is None, "Should return None when database doesn't exist"


if __name__ == "__main__":
    # This will run the tests if the file is executed directly
    setup_module()
    import pytest

    pytest.main(["-v", __file__])
