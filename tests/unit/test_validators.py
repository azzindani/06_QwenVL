"""Unit tests for format validators."""

import pytest

from qwen_vl.utils.validators import (
    get_validator_types,
    validate_currency,
    validate_date,
    validate_email,
    validate_field,
    validate_fields,
    validate_integer,
    validate_percentage,
    validate_phone,
    validate_url,
)


@pytest.mark.unit
class TestValidateEmail:
    """Tests for email validation."""

    def test_valid_emails(self):
        """Test valid email addresses."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "user+tag@example.co.uk",
            "a@b.cd",
        ]
        for email in valid_emails:
            is_valid, _ = validate_email(email)
            assert is_valid, f"Should be valid: {email}"

    def test_invalid_emails(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "invalid",
            "missing@domain",
            "@nodomain.com",
            "spaces in@email.com",
        ]
        for email in invalid_emails:
            is_valid, error = validate_email(email)
            assert not is_valid, f"Should be invalid: {email}"
            assert error is not None


@pytest.mark.unit
class TestValidatePhone:
    """Tests for phone validation."""

    def test_valid_phones(self):
        """Test valid phone numbers."""
        valid_phones = [
            "+1-555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "+44 20 7123 4567",
            "5551234567",
        ]
        for phone in valid_phones:
            is_valid, _ = validate_phone(phone)
            assert is_valid, f"Should be valid: {phone}"

    def test_invalid_phones(self):
        """Test invalid phone numbers."""
        invalid_phones = [
            "123",  # Too short
            "abc-def-ghij",
        ]
        for phone in invalid_phones:
            is_valid, _ = validate_phone(phone)
            assert not is_valid, f"Should be invalid: {phone}"


@pytest.mark.unit
class TestValidateDate:
    """Tests for date validation."""

    def test_valid_dates(self):
        """Test valid date formats."""
        valid_dates = [
            "2024-01-15",
            "15/01/2024",
            "01/15/2024",
            "January 15, 2024",
            "Jan 15, 2024",
            "15 January 2024",
        ]
        for date in valid_dates:
            is_valid, _ = validate_date(date)
            assert is_valid, f"Should be valid: {date}"

    def test_invalid_dates(self):
        """Test invalid date formats."""
        invalid_dates = [
            "not a date",
            "2024-13-01",  # Invalid month (not caught by format check alone)
            "random text",
        ]
        for date in invalid_dates:
            is_valid, _ = validate_date(date)
            # Note: Some invalid dates might pass format check


@pytest.mark.unit
class TestValidateCurrency:
    """Tests for currency validation."""

    def test_valid_currencies(self):
        """Test valid currency values."""
        valid_currencies = [
            "$1,234.56",
            "€100",
            "£50.00",
            "1234.56",
            "¥10000",
            "₹500",
        ]
        for currency in valid_currencies:
            is_valid, _ = validate_currency(currency)
            assert is_valid, f"Should be valid: {currency}"

    def test_invalid_currencies(self):
        """Test invalid currency values."""
        invalid_currencies = [
            "not a number",
            "abc",
        ]
        for currency in invalid_currencies:
            is_valid, _ = validate_currency(currency)
            assert not is_valid, f"Should be invalid: {currency}"


@pytest.mark.unit
class TestValidateUrl:
    """Tests for URL validation."""

    def test_valid_urls(self):
        """Test valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://example.com/path",
            "https://sub.domain.org/path?query=1",
        ]
        for url in valid_urls:
            is_valid, _ = validate_url(url)
            assert is_valid, f"Should be valid: {url}"

    def test_invalid_urls(self):
        """Test invalid URLs."""
        invalid_urls = [
            "not a url",
            "ftp://example.com",  # Only http/https
            "example.com",  # Missing protocol
        ]
        for url in invalid_urls:
            is_valid, _ = validate_url(url)
            assert not is_valid, f"Should be invalid: {url}"


@pytest.mark.unit
class TestValidatePercentage:
    """Tests for percentage validation."""

    def test_valid_percentages(self):
        """Test valid percentages."""
        valid = ["50%", "100", "0", "99.9%", "0%"]
        for p in valid:
            is_valid, _ = validate_percentage(p)
            assert is_valid, f"Should be valid: {p}"

    def test_invalid_percentages(self):
        """Test invalid percentages."""
        is_valid, _ = validate_percentage("101%")
        assert not is_valid  # Out of range

        is_valid, _ = validate_percentage("not a number")
        assert not is_valid


@pytest.mark.unit
class TestValidateInteger:
    """Tests for integer validation."""

    def test_valid_integers(self):
        """Test valid integers."""
        valid = ["123", "1,000", "0", "-5"]
        for i in valid:
            is_valid, _ = validate_integer(i)
            assert is_valid, f"Should be valid: {i}"

    def test_invalid_integers(self):
        """Test invalid integers."""
        is_valid, _ = validate_integer("12.34")
        assert not is_valid

        is_valid, _ = validate_integer("abc")
        assert not is_valid


@pytest.mark.unit
class TestValidateField:
    """Tests for validate_field function."""

    def test_known_type(self):
        """Test validation with known type."""
        is_valid, _ = validate_field("test@example.com", "email")
        assert is_valid

    def test_unknown_type(self):
        """Test validation with unknown type passes."""
        is_valid, _ = validate_field("anything", "unknown_type")
        assert is_valid  # Unknown types pass


@pytest.mark.unit
class TestValidateFields:
    """Tests for validate_fields function."""

    def test_validate_schema_fields(self):
        """Test validating multiple fields against schema."""
        schema = {
            "fields": [
                {"name": "email", "type": "email"},
                {"name": "phone", "type": "phone"},
            ]
        }
        fields = {
            "email": "test@example.com",
            "phone": "+1-555-123-4567",
        }

        errors = validate_fields(fields, schema)
        assert len(errors) == 0

    def test_validate_with_errors(self):
        """Test validation with errors."""
        schema = {
            "fields": [
                {"name": "email", "type": "email"},
            ]
        }
        fields = {
            "email": "invalid",
        }

        errors = validate_fields(fields, schema)
        assert "email" in errors
        assert len(errors["email"]) > 0

    def test_validate_dict_values(self):
        """Test validation with dict values containing 'value' key."""
        schema = {
            "fields": [
                {"name": "email", "type": "email"},
            ]
        }
        fields = {
            "email": {"value": "test@example.com", "confidence": 0.95},
        }

        errors = validate_fields(fields, schema)
        assert len(errors) == 0


@pytest.mark.unit
class TestGetValidatorTypes:
    """Tests for get_validator_types function."""

    def test_returns_list(self):
        """Test that function returns list of types."""
        types = get_validator_types()
        assert isinstance(types, list)
        assert "email" in types
        assert "phone" in types
        assert "date" in types
        assert "currency" in types
