"""Unit tests for Phase 3 task handlers."""

import pytest

from qwen_vl.tasks import TaskType, list_handlers
from qwen_vl.tasks.form import FormHandler
from qwen_vl.tasks.invoice import InvoiceHandler
from qwen_vl.tasks.contract import ContractHandler


@pytest.mark.unit
class TestPhase3Registration:
    """Tests for Phase 3 handler registration."""

    def test_form_handler_registered(self):
        """Test that FormHandler is registered."""
        handlers = list_handlers()
        assert TaskType.FORM in handlers

    def test_invoice_handler_registered(self):
        """Test that InvoiceHandler is registered."""
        handlers = list_handlers()
        assert TaskType.INVOICE in handlers

    def test_contract_handler_registered(self):
        """Test that ContractHandler is registered."""
        handlers = list_handlers()
        assert TaskType.CONTRACT in handlers


@pytest.mark.unit
class TestFormHandler:
    """Tests for FormHandler."""

    def test_form_prompt_generation(self):
        """Test form prompt includes all sections."""
        handler = FormHandler.__new__(FormHandler)
        prompt = handler._build_form_prompt(
            extract_signatures=True,
            extract_checkboxes=True,
        )

        assert "Key-Value Pairs" in prompt
        assert "Checkboxes" in prompt
        assert "Signatures" in prompt
        assert "json" in prompt

    def test_form_prompt_without_signatures(self):
        """Test prompt without signature extraction."""
        handler = FormHandler.__new__(FormHandler)
        prompt = handler._build_form_prompt(
            extract_signatures=False,
            extract_checkboxes=True,
        )

        assert "Checkboxes" in prompt
        assert "Signatures" not in prompt

    def test_form_prompt_without_checkboxes(self):
        """Test prompt without checkbox extraction."""
        handler = FormHandler.__new__(FormHandler)
        prompt = handler._build_form_prompt(
            extract_signatures=True,
            extract_checkboxes=False,
        )

        assert "Signatures" in prompt
        assert "Checkboxes" not in prompt

    def test_form_system_prompt(self):
        """Test system prompt content."""
        handler = FormHandler.__new__(FormHandler)
        prompt = handler.system_prompt

        assert "form" in prompt.lower()
        assert "key-value" in prompt.lower()

    def test_form_task_type(self):
        """Test task type property."""
        handler = FormHandler.__new__(FormHandler)
        assert handler.task_type == TaskType.FORM


@pytest.mark.unit
class TestInvoiceHandler:
    """Tests for InvoiceHandler."""

    def test_invoice_prompt_generation(self):
        """Test invoice prompt includes key sections."""
        handler = InvoiceHandler.__new__(InvoiceHandler)
        prompt = handler._build_invoice_prompt("invoice")

        assert "invoice" in prompt.lower()
        assert "vendor" in prompt.lower()
        assert "line_items" in prompt
        assert "total" in prompt.lower()

    def test_receipt_prompt_generation(self):
        """Test receipt prompt."""
        handler = InvoiceHandler.__new__(InvoiceHandler)
        prompt = handler._build_invoice_prompt("receipt")

        assert "receipt" in prompt.lower()

    def test_invoice_validation_valid(self):
        """Test validation with valid data."""
        handler = InvoiceHandler.__new__(InvoiceHandler)

        line_items = [
            {"amount": 10.0},
            {"amount": 20.0},
        ]
        summary = {
            "subtotal": 30.0,
            "tax": 3.0,
            "discount": 0,
            "total": 33.0,
        }

        result = handler._validate_invoice(line_items, summary)
        assert result["is_valid"]
        assert len(result["errors"]) == 0

    def test_invoice_validation_subtotal_mismatch(self):
        """Test validation catches subtotal mismatch."""
        handler = InvoiceHandler.__new__(InvoiceHandler)

        line_items = [
            {"amount": 10.0},
            {"amount": 20.0},
        ]
        summary = {
            "subtotal": 50.0,  # Wrong
            "tax": 5.0,
            "discount": 0,
            "total": 55.0,
        }

        result = handler._validate_invoice(line_items, summary)
        assert not result["is_valid"]
        assert any("Subtotal" in e for e in result["errors"])

    def test_invoice_validation_total_mismatch(self):
        """Test validation catches total mismatch."""
        handler = InvoiceHandler.__new__(InvoiceHandler)

        line_items = [{"amount": 100.0}]
        summary = {
            "subtotal": 100.0,
            "tax": 10.0,
            "discount": 0,
            "total": 50.0,  # Wrong
        }

        result = handler._validate_invoice(line_items, summary)
        assert not result["is_valid"]
        assert any("Total" in e for e in result["errors"])

    def test_invoice_validation_line_item_warning(self):
        """Test validation warns about line item calculation."""
        handler = InvoiceHandler.__new__(InvoiceHandler)

        line_items = [
            {
                "quantity": 2,
                "unit_price": 10.0,
                "amount": 15.0,  # Should be 20
            }
        ]
        summary = {
            "subtotal": 15.0,
            "tax": 0,
            "discount": 0,
            "total": 15.0,
        }

        result = handler._validate_invoice(line_items, summary)
        assert len(result["warnings"]) > 0

    def test_invoice_system_prompt(self):
        """Test system prompt content."""
        handler = InvoiceHandler.__new__(InvoiceHandler)
        prompt = handler.system_prompt

        assert "invoice" in prompt.lower()
        assert "receipt" in prompt.lower()

    def test_invoice_task_type(self):
        """Test task type property."""
        handler = InvoiceHandler.__new__(InvoiceHandler)
        assert handler.task_type == TaskType.INVOICE


@pytest.mark.unit
class TestContractHandler:
    """Tests for ContractHandler."""

    def test_contract_prompt_generation(self):
        """Test contract prompt includes key sections."""
        handler = ContractHandler.__new__(ContractHandler)
        prompt = handler._build_contract_prompt(
            extract_clauses=True,
            extract_obligations=True,
        )

        assert "Parties" in prompt
        assert "Dates" in prompt
        assert "Clauses" in prompt
        assert "Obligations" in prompt
        assert "Key Terms" in prompt

    def test_contract_prompt_without_clauses(self):
        """Test prompt without clause extraction."""
        handler = ContractHandler.__new__(ContractHandler)
        prompt = handler._build_contract_prompt(
            extract_clauses=False,
            extract_obligations=True,
        )

        assert "Obligations" in prompt
        # Clauses section header should not appear
        assert "3. **Clauses**" not in prompt

    def test_contract_prompt_without_obligations(self):
        """Test prompt without obligation extraction."""
        handler = ContractHandler.__new__(ContractHandler)
        prompt = handler._build_contract_prompt(
            extract_clauses=True,
            extract_obligations=False,
        )

        assert "Clauses" in prompt
        assert "4. **Obligations**" not in prompt

    def test_contract_system_prompt(self):
        """Test system prompt content."""
        handler = ContractHandler.__new__(ContractHandler)
        prompt = handler.system_prompt

        assert "contract" in prompt.lower()
        assert "parties" in prompt.lower()

    def test_contract_task_type(self):
        """Test task type property."""
        handler = ContractHandler.__new__(ContractHandler)
        assert handler.task_type == TaskType.CONTRACT
