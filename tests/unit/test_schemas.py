"""Unit tests for schemas and presets."""

import pytest

from qwen_vl.tasks import ENTITY_TYPES, PRESET_SCHEMAS, TaskType
from qwen_vl.tasks.field_extraction import FieldExtractionHandler
from qwen_vl.tasks.ner import NERHandler


@pytest.mark.unit
class TestPresetSchemas:
    """Tests for preset schemas."""

    def test_preset_schemas_exist(self):
        """Test that expected presets exist."""
        assert "invoice" in PRESET_SCHEMAS
        assert "receipt" in PRESET_SCHEMAS
        assert "id_card" in PRESET_SCHEMAS
        assert "business_card" in PRESET_SCHEMAS

    def test_invoice_schema(self):
        """Test invoice schema has expected fields."""
        schema = PRESET_SCHEMAS["invoice"]
        field_names = [f["name"] for f in schema["fields"]]

        assert "invoice_number" in field_names
        assert "date" in field_names
        assert "total" in field_names
        assert "vendor_name" in field_names

    def test_receipt_schema(self):
        """Test receipt schema has expected fields."""
        schema = PRESET_SCHEMAS["receipt"]
        field_names = [f["name"] for f in schema["fields"]]

        assert "store_name" in field_names
        assert "date" in field_names
        assert "total" in field_names
        assert "items" in field_names

    def test_id_card_schema(self):
        """Test ID card schema has expected fields."""
        schema = PRESET_SCHEMAS["id_card"]
        field_names = [f["name"] for f in schema["fields"]]

        assert "full_name" in field_names
        assert "date_of_birth" in field_names
        assert "id_number" in field_names

    def test_business_card_schema(self):
        """Test business card schema has expected fields."""
        schema = PRESET_SCHEMAS["business_card"]
        field_names = [f["name"] for f in schema["fields"]]

        assert "name" in field_names
        assert "email" in field_names
        assert "phone" in field_names
        assert "company" in field_names

    def test_field_structure(self):
        """Test that fields have required keys."""
        for preset_name, schema in PRESET_SCHEMAS.items():
            for field in schema["fields"]:
                assert "name" in field, f"Missing 'name' in {preset_name}"
                assert "type" in field, f"Missing 'type' in {preset_name}"
                assert "description" in field, f"Missing 'description' in {preset_name}"


@pytest.mark.unit
class TestFieldExtractionHandler:
    """Tests for FieldExtractionHandler static methods."""

    def test_list_presets(self):
        """Test listing available presets."""
        presets = FieldExtractionHandler.list_presets()
        assert isinstance(presets, list)
        assert "invoice" in presets
        assert "receipt" in presets

    def test_get_preset_schema(self):
        """Test getting preset schema by name."""
        schema = FieldExtractionHandler.get_preset_schema("invoice")
        assert schema is not None
        assert "fields" in schema

    def test_get_invalid_preset(self):
        """Test getting non-existent preset returns None."""
        schema = FieldExtractionHandler.get_preset_schema("invalid")
        assert schema is None


@pytest.mark.unit
class TestEntityTypes:
    """Tests for NER entity types."""

    def test_entity_types_exist(self):
        """Test that expected entity types exist."""
        assert "PERSON" in ENTITY_TYPES
        assert "ORGANIZATION" in ENTITY_TYPES
        assert "LOCATION" in ENTITY_TYPES
        assert "DATE" in ENTITY_TYPES
        assert "MONEY" in ENTITY_TYPES
        assert "EMAIL" in ENTITY_TYPES
        assert "PHONE" in ENTITY_TYPES

    def test_entity_types_have_descriptions(self):
        """Test that all entity types have descriptions."""
        for entity_type, description in ENTITY_TYPES.items():
            assert isinstance(description, str)
            assert len(description) > 0


@pytest.mark.unit
class TestNERHandler:
    """Tests for NERHandler static methods."""

    def test_list_entity_types(self):
        """Test listing entity types."""
        types = NERHandler.list_entity_types()
        assert isinstance(types, dict)
        assert "PERSON" in types


@pytest.mark.unit
class TestTaskTypeRegistration:
    """Tests for task type registration."""

    def test_new_handlers_registered(self):
        """Test that Phase 2 handlers are registered."""
        from qwen_vl.tasks import list_handlers

        handlers = list_handlers()
        assert TaskType.TABLE in handlers
        assert TaskType.FIELD_EXTRACTION in handlers
        assert TaskType.NER in handlers


@pytest.mark.unit
class TestSchemaValidation:
    """Tests for schema structure validation."""

    def test_custom_schema_format(self):
        """Test custom schema format works."""
        custom_schema = {
            "fields": [
                {"name": "custom_field", "type": "string", "description": "Custom"},
            ]
        }
        # Should not raise
        field_names = [f["name"] for f in custom_schema["fields"]]
        assert "custom_field" in field_names

    def test_empty_schema(self):
        """Test empty schema is valid."""
        empty_schema = {"fields": []}
        assert len(empty_schema["fields"]) == 0
