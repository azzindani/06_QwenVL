"""Unit tests for cross-field validation utilities."""

import pytest

from qwen_vl.utils.cross_validation import (
    validate_date_consistency,
    validate_total_calculation,
    validate_required_fields,
    validate_field_dependencies,
    validate_cross_references,
)


@pytest.mark.unit
class TestDateConsistency:
    """Tests for date consistency validation."""

    def test_valid_date_order(self):
        """Test valid date ordering passes."""
        dates = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }
        errors = validate_date_consistency(dates)
        assert len(errors) == 0

    def test_invalid_date_order(self):
        """Test invalid date ordering fails."""
        dates = {
            "start_date": "2024-12-31",
            "end_date": "2024-01-01",
        }
        errors = validate_date_consistency(dates)
        assert len(errors) > 0
        assert "before" in errors[0].lower()

    def test_invoice_dates(self):
        """Test invoice date validation."""
        dates = {
            "invoice_date": "2024-01-15",
            "due_date": "2024-02-15",
        }
        errors = validate_date_consistency(dates)
        assert len(errors) == 0

    def test_effective_termination_dates(self):
        """Test contract date validation."""
        dates = {
            "effective_date": "2024-01-01",
            "termination_date": "2025-01-01",
        }
        errors = validate_date_consistency(dates)
        assert len(errors) == 0

    def test_custom_rules(self):
        """Test custom date rules."""
        dates = {
            "shipped": "2024-01-15",
            "delivered": "2024-01-10",
        }
        rules = [{"before": "shipped", "after": "delivered"}]
        errors = validate_date_consistency(dates, rules)
        assert len(errors) > 0

    def test_missing_dates_ignored(self):
        """Test that missing dates are ignored."""
        dates = {
            "start_date": "2024-01-01",
        }
        errors = validate_date_consistency(dates)
        assert len(errors) == 0

    def test_different_date_formats(self):
        """Test various date formats."""
        dates = {
            "start_date": "January 15, 2024",
            "end_date": "2024-12-31",
        }
        errors = validate_date_consistency(dates)
        assert len(errors) == 0


@pytest.mark.unit
class TestTotalCalculation:
    """Tests for total calculation validation."""

    def test_valid_totals(self):
        """Test valid total calculation."""
        line_items = [
            {"amount": 10.0},
            {"amount": 20.0},
            {"amount": 30.0},
        ]
        summary = {
            "subtotal": 60.0,
            "tax": 6.0,
            "discount": 0,
            "total": 66.0,
        }
        result = validate_total_calculation(line_items, summary)
        assert result["is_valid"]
        assert result["calculated"]["subtotal"] == 60.0

    def test_subtotal_mismatch(self):
        """Test subtotal mismatch detection."""
        line_items = [
            {"amount": 10.0},
            {"amount": 20.0},
        ]
        summary = {
            "subtotal": 100.0,  # Wrong
            "tax": 10.0,
            "discount": 0,
            "total": 110.0,
        }
        result = validate_total_calculation(line_items, summary)
        assert not result["is_valid"]
        assert any("Subtotal" in e for e in result["errors"])

    def test_total_mismatch(self):
        """Test total mismatch detection."""
        line_items = [{"amount": 100.0}]
        summary = {
            "subtotal": 100.0,
            "tax": 10.0,
            "discount": 5.0,
            "total": 200.0,  # Should be 105
        }
        result = validate_total_calculation(line_items, summary)
        assert not result["is_valid"]

    def test_line_item_calculation(self):
        """Test line item qty * price validation."""
        line_items = [
            {
                "quantity": 5,
                "unit_price": 10.0,
                "amount": 50.0,
            }
        ]
        summary = {
            "subtotal": 50.0,
            "tax": 0,
            "discount": 0,
            "total": 50.0,
        }
        result = validate_total_calculation(line_items, summary)
        assert result["is_valid"]

    def test_line_item_calculation_error(self):
        """Test line item calculation error detection."""
        line_items = [
            {
                "quantity": 5,
                "unit_price": 10.0,
                "amount": 40.0,  # Should be 50
            }
        ]
        summary = {
            "subtotal": 40.0,
            "tax": 0,
            "discount": 0,
            "total": 40.0,
        }
        result = validate_total_calculation(line_items, summary)
        assert len(result["errors"]) > 0

    def test_with_discount(self):
        """Test total with discount."""
        line_items = [{"amount": 100.0}]
        summary = {
            "subtotal": 100.0,
            "tax": 10.0,
            "discount": 20.0,
            "total": 90.0,
        }
        result = validate_total_calculation(line_items, summary)
        assert result["is_valid"]

    def test_tolerance(self):
        """Test floating point tolerance."""
        line_items = [{"amount": 33.33}, {"amount": 33.33}, {"amount": 33.34}]
        summary = {
            "subtotal": 100.0,
            "tax": 0,
            "discount": 0,
            "total": 100.0,
        }
        result = validate_total_calculation(line_items, summary)
        assert result["is_valid"]

    def test_currency_strings(self):
        """Test handling currency string values."""
        line_items = [{"amount": "$50.00"}]
        summary = {
            "subtotal": "50.00",
            "tax": "$5.00",
            "discount": "0",
            "total": "$55.00",
        }
        result = validate_total_calculation(line_items, summary)
        assert result["is_valid"]


@pytest.mark.unit
class TestRequiredFields:
    """Tests for required field validation."""

    def test_all_present(self):
        """Test all required fields present."""
        data = {
            "name": "John",
            "email": "john@example.com",
        }
        missing = validate_required_fields(data, ["name", "email"])
        assert len(missing) == 0

    def test_missing_field(self):
        """Test missing field detection."""
        data = {"name": "John"}
        missing = validate_required_fields(data, ["name", "email"])
        assert "email" in missing

    def test_empty_value(self):
        """Test empty value counts as missing."""
        data = {"name": "", "email": "john@example.com"}
        missing = validate_required_fields(data, ["name", "email"])
        assert "name" in missing

    def test_nested_fields(self):
        """Test nested field validation."""
        data = {
            "header": {
                "vendor_name": "Company",
            }
        }
        missing = validate_required_fields(data, ["header.vendor_name", "header.date"])
        assert "header.date" in missing
        assert "header.vendor_name" not in missing

    def test_empty_list(self):
        """Test empty list counts as missing."""
        data = {"items": []}
        missing = validate_required_fields(data, ["items"])
        assert "items" in missing


@pytest.mark.unit
class TestFieldDependencies:
    """Tests for field dependency validation."""

    def test_satisfied_dependency(self):
        """Test satisfied dependency passes."""
        data = {
            "discount": 10,
            "discount_reason": "Holiday sale",
        }
        dependencies = [
            {"if_field": "discount", "then_required": ["discount_reason"]}
        ]
        errors = validate_field_dependencies(data, dependencies)
        assert len(errors) == 0

    def test_unsatisfied_dependency(self):
        """Test unsatisfied dependency fails."""
        data = {
            "discount": 10,
        }
        dependencies = [
            {"if_field": "discount", "then_required": ["discount_reason"]}
        ]
        errors = validate_field_dependencies(data, dependencies)
        assert len(errors) > 0
        assert "discount_reason" in errors[0]

    def test_no_trigger(self):
        """Test no trigger means no requirement."""
        data = {}
        dependencies = [
            {"if_field": "discount", "then_required": ["discount_reason"]}
        ]
        errors = validate_field_dependencies(data, dependencies)
        assert len(errors) == 0

    def test_multiple_dependencies(self):
        """Test multiple dependent fields."""
        data = {
            "refund": True,
        }
        dependencies = [
            {"if_field": "refund", "then_required": ["refund_reason", "refund_amount"]}
        ]
        errors = validate_field_dependencies(data, dependencies)
        assert len(errors) == 2


@pytest.mark.unit
class TestCrossReferences:
    """Tests for cross-reference validation."""

    def test_matching_values(self):
        """Test matching values pass."""
        primary = {"invoice_total": 100.0}
        secondary = {"payment_amount": 100.0}
        errors = validate_cross_references(
            primary, secondary, [("invoice_total", "payment_amount")]
        )
        assert len(errors) == 0

    def test_mismatched_values(self):
        """Test mismatched values fail."""
        primary = {"invoice_total": 100.0}
        secondary = {"payment_amount": 50.0}
        errors = validate_cross_references(
            primary, secondary, [("invoice_total", "payment_amount")]
        )
        assert len(errors) > 0

    def test_case_insensitive(self):
        """Test case insensitive comparison."""
        primary = {"name": "JOHN DOE"}
        secondary = {"customer": "John Doe"}
        errors = validate_cross_references(
            primary, secondary, [("name", "customer")]
        )
        assert len(errors) == 0

    def test_missing_value_ignored(self):
        """Test missing values are ignored."""
        primary = {"name": "John"}
        secondary = {}
        errors = validate_cross_references(
            primary, secondary, [("name", "customer")]
        )
        assert len(errors) == 0
