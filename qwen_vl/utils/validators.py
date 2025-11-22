"""Format validators for extracted data."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def validate_email(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, value):
        return True, None
    return False, f"Invalid email format: {value}"


def validate_phone(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate phone number format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Remove common separators
    cleaned = re.sub(r'[\s\-\.\(\)]', '', value)
    # Check if it's mostly digits
    if re.match(r'^\+?\d{7,15}$', cleaned):
        return True, None
    return False, f"Invalid phone format: {value}"


def validate_date(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate date format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Common date formats
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]

    for fmt in formats:
        try:
            datetime.strptime(value.strip(), fmt)
            return True, None
        except ValueError:
            continue

    return False, f"Invalid date format: {value}"


def validate_currency(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate currency/monetary value format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Remove currency symbols and spaces
    cleaned = re.sub(r'[$€£¥₹\s,]', '', value)

    # Check if it's a valid number
    try:
        float(cleaned)
        return True, None
    except ValueError:
        return False, f"Invalid currency format: {value}"


def validate_url(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if re.match(pattern, value, re.IGNORECASE):
        return True, None
    return False, f"Invalid URL format: {value}"


def validate_percentage(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate percentage format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    cleaned = value.strip().rstrip('%')
    try:
        num = float(cleaned)
        if 0 <= num <= 100:
            return True, None
        return False, f"Percentage out of range: {value}"
    except ValueError:
        return False, f"Invalid percentage format: {value}"


def validate_integer(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate integer format.

    Returns:
        Tuple of (is_valid, error_message)
    """
    cleaned = re.sub(r'[,\s]', '', value)
    try:
        int(cleaned)
        return True, None
    except ValueError:
        return False, f"Invalid integer format: {value}"


# Validator registry
VALIDATORS = {
    "email": validate_email,
    "phone": validate_phone,
    "date": validate_date,
    "currency": validate_currency,
    "url": validate_url,
    "percentage": validate_percentage,
    "integer": validate_integer,
}


def validate_field(value: str, field_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a field value against its type.

    Args:
        value: Field value to validate
        field_type: Type of field

    Returns:
        Tuple of (is_valid, error_message)
    """
    if field_type not in VALIDATORS:
        return True, None  # No validation for unknown types

    return VALIDATORS[field_type](value)


def validate_fields(fields: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate multiple fields against a schema.

    Args:
        fields: Dict of field_name -> value
        schema: Schema with 'fields' list containing field definitions

    Returns:
        Dict of field_name -> list of error messages (empty if valid)
    """
    errors = {}

    schema_fields = {f["name"]: f for f in schema.get("fields", [])}

    for field_name, field_value in fields.items():
        if field_name not in schema_fields:
            continue

        field_def = schema_fields[field_name]
        field_type = field_def.get("type", "string")

        # Get actual value if it's a dict with 'value' key
        if isinstance(field_value, dict):
            actual_value = field_value.get("value", "")
        else:
            actual_value = str(field_value)

        if actual_value:  # Only validate non-empty values
            is_valid, error = validate_field(actual_value, field_type)
            if not is_valid:
                if field_name not in errors:
                    errors[field_name] = []
                errors[field_name].append(error)

    return errors


def get_validator_types() -> List[str]:
    """Get list of available validator types."""
    return list(VALIDATORS.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("VALIDATORS TEST")
    print("=" * 60)

    # Test email
    valid, err = validate_email("test@example.com")
    print(f"  Email 'test@example.com': {valid}")
    assert valid

    valid, err = validate_email("invalid")
    print(f"  Email 'invalid': {valid}")
    assert not valid

    # Test phone
    valid, err = validate_phone("+1-555-123-4567")
    print(f"  Phone '+1-555-123-4567': {valid}")
    assert valid

    # Test date
    valid, err = validate_date("2024-01-15")
    print(f"  Date '2024-01-15': {valid}")
    assert valid

    valid, err = validate_date("January 15, 2024")
    print(f"  Date 'January 15, 2024': {valid}")
    assert valid

    # Test currency
    valid, err = validate_currency("$1,234.56")
    print(f"  Currency '$1,234.56': {valid}")
    assert valid

    # Test URL
    valid, err = validate_url("https://example.com")
    print(f"  URL 'https://example.com': {valid}")
    assert valid

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
