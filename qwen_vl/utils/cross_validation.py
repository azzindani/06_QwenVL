"""Cross-field validation utilities for document data."""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import re


def validate_date_consistency(
    dates: Dict[str, str],
    rules: Optional[List[Dict[str, str]]] = None,
) -> List[str]:
    """
    Validate date consistency across fields.

    Args:
        dates: Dict of field_name -> date_string
        rules: List of rules like {"before": "field1", "after": "field2"}

    Returns:
        List of error messages
    """
    errors = []

    # Default rules if not provided
    if rules is None:
        rules = [
            {"before": "start_date", "after": "end_date"},
            {"before": "effective_date", "after": "termination_date"},
            {"before": "invoice_date", "after": "due_date"},
            {"before": "order_date", "after": "delivery_date"},
        ]

    # Parse dates
    parsed = {}
    for field, value in dates.items():
        if not value:
            continue
        parsed_date = _parse_date(value)
        if parsed_date:
            parsed[field] = parsed_date

    # Check rules
    for rule in rules:
        before_field = rule.get("before", "")
        after_field = rule.get("after", "")

        if before_field in parsed and after_field in parsed:
            if parsed[before_field] > parsed[after_field]:
                errors.append(
                    f"{before_field} ({dates[before_field]}) should be before "
                    f"{after_field} ({dates[after_field]})"
                )

    return errors


def validate_total_calculation(
    line_items: List[Dict[str, Any]],
    summary: Dict[str, Any],
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    """
    Validate that totals match calculated values.

    Args:
        line_items: List of line items with amount/quantity/unit_price
        summary: Summary with subtotal/tax/discount/total
        tolerance: Acceptable difference for floating point comparison

    Returns:
        Dict with is_valid, errors, and calculated values
    """
    errors = []
    calculated = {}

    # Calculate subtotal from line items
    item_subtotal = 0.0
    for i, item in enumerate(line_items):
        amount = _to_float(item.get("amount", 0))
        qty = _to_float(item.get("quantity", 1))
        price = _to_float(item.get("unit_price", 0))

        if amount > 0:
            item_subtotal += amount
        elif qty > 0 and price > 0:
            item_subtotal += qty * price

        # Validate individual line item
        if qty > 0 and price > 0 and amount > 0:
            expected = qty * price
            if abs(expected - amount) > tolerance:
                errors.append(
                    f"Line {i+1}: quantity * unit_price ({expected:.2f}) != amount ({amount:.2f})"
                )

    calculated["subtotal"] = item_subtotal

    # Get summary values
    reported_subtotal = _to_float(summary.get("subtotal", 0))
    tax = _to_float(summary.get("tax", 0))
    discount = _to_float(summary.get("discount", 0))
    total = _to_float(summary.get("total", 0))

    # Validate subtotal
    if item_subtotal > 0 and reported_subtotal > 0:
        if abs(item_subtotal - reported_subtotal) > tolerance:
            errors.append(
                f"Subtotal mismatch: calculated {item_subtotal:.2f}, "
                f"reported {reported_subtotal:.2f}"
            )

    # Validate total = subtotal + tax - discount
    base = reported_subtotal if reported_subtotal > 0 else item_subtotal
    expected_total = base + tax - discount
    calculated["expected_total"] = expected_total

    if total > 0 and abs(expected_total - total) > tolerance:
        errors.append(
            f"Total mismatch: subtotal ({base:.2f}) + tax ({tax:.2f}) - "
            f"discount ({discount:.2f}) = {expected_total:.2f}, "
            f"but total is {total:.2f}"
        )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "calculated": calculated,
    }


def validate_required_fields(
    data: Dict[str, Any],
    required: List[str],
) -> List[str]:
    """
    Validate that required fields are present and non-empty.

    Args:
        data: Data dict to validate
        required: List of required field names (supports dot notation)

    Returns:
        List of missing/empty field names
    """
    missing = []

    for field in required:
        value = _get_nested(data, field)
        if value is None or value == "" or value == []:
            missing.append(field)

    return missing


def validate_field_dependencies(
    data: Dict[str, Any],
    dependencies: List[Dict[str, Any]],
) -> List[str]:
    """
    Validate field dependencies.

    Args:
        data: Data dict to validate
        dependencies: List of dependency rules

    Returns:
        List of error messages

    Example dependency:
        {"if_field": "discount", "then_required": ["discount_reason"]}
    """
    errors = []

    for dep in dependencies:
        if_field = dep.get("if_field", "")
        then_required = dep.get("then_required", [])

        # Check if trigger field has value
        trigger_value = _get_nested(data, if_field)
        if trigger_value and trigger_value != 0:
            # Check required fields
            for req_field in then_required:
                value = _get_nested(data, req_field)
                if value is None or value == "":
                    errors.append(
                        f"{req_field} is required when {if_field} is present"
                    )

    return errors


def validate_cross_references(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    field_pairs: List[Tuple[str, str]],
) -> List[str]:
    """
    Validate that fields match between two data sources.

    Args:
        primary: Primary data dict
        secondary: Secondary data dict
        field_pairs: List of (primary_field, secondary_field) tuples

    Returns:
        List of mismatch errors
    """
    errors = []

    for primary_field, secondary_field in field_pairs:
        primary_value = _get_nested(primary, primary_field)
        secondary_value = _get_nested(secondary, secondary_field)

        if primary_value and secondary_value:
            # Normalize for comparison
            p_norm = _normalize_value(primary_value)
            s_norm = _normalize_value(secondary_value)

            if p_norm != s_norm:
                errors.append(
                    f"Mismatch: {primary_field}='{primary_value}' vs "
                    f"{secondary_field}='{secondary_value}'"
                )

    return errors


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime."""
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def _to_float(value: Any) -> float:
    """Convert value to float."""
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$€£¥₹,\s]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    return 0.0


def _get_nested(data: Dict[str, Any], path: str) -> Any:
    """Get nested value using dot notation."""
    keys = path.split(".")
    value = data

    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None

        if value is None:
            return None

    return value


def _normalize_value(value: Any) -> str:
    """Normalize value for comparison."""
    if value is None:
        return ""

    s = str(value).lower().strip()
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s)
    return s


if __name__ == "__main__":
    print("=" * 60)
    print("CROSS-FIELD VALIDATION TEST")
    print("=" * 60)

    # Test total calculation
    line_items = [
        {"description": "Item 1", "quantity": 2, "unit_price": 10.0, "amount": 20.0},
        {"description": "Item 2", "quantity": 1, "unit_price": 15.0, "amount": 15.0},
    ]
    summary = {"subtotal": 35.0, "tax": 3.5, "discount": 0, "total": 38.5}

    result = validate_total_calculation(line_items, summary)
    print(f"  Total validation: {result['is_valid']}")
    assert result["is_valid"]

    # Test date consistency
    dates = {"start_date": "2024-01-01", "end_date": "2024-12-31"}
    errors = validate_date_consistency(dates)
    print(f"  Date validation errors: {len(errors)}")
    assert len(errors) == 0

    # Test invalid dates
    dates = {"start_date": "2024-12-31", "end_date": "2024-01-01"}
    errors = validate_date_consistency(dates)
    print(f"  Invalid date errors: {len(errors)}")
    assert len(errors) > 0

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
