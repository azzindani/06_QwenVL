"""Unit tests for API schema generation."""

import pytest

from qwen_vl.api.schemas import (
    schema_to_pydantic,
    generate_extraction_models,
    ExtractionResult,
    BatchJobStatus,
)


@pytest.mark.unit
class TestSchemaToPydantic:
    """Tests for schema to Pydantic model conversion."""

    def test_basic_schema(self):
        """Test converting basic schema."""
        schema = {
            "fields": [
                {"name": "vendor_name", "type": "text", "required": True},
                {"name": "total", "type": "number", "required": True},
            ]
        }

        Model = schema_to_pydantic("TestModel", schema)

        assert Model.__name__ == "TestModel"
        assert "vendor_name" in Model.model_fields
        assert "total" in Model.model_fields

    def test_optional_fields(self):
        """Test optional fields in schema."""
        schema = {
            "fields": [
                {"name": "required_field", "type": "text", "required": True},
                {"name": "optional_field", "type": "text", "required": False},
            ]
        }

        Model = schema_to_pydantic("OptionalModel", schema)

        # Create instance with only required field
        instance = Model(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field is None

    def test_type_mapping(self):
        """Test field type to Python type mapping."""
        schema = {
            "fields": [
                {"name": "text_field", "type": "text", "required": True},
                {"name": "number_field", "type": "number", "required": True},
                {"name": "int_field", "type": "integer", "required": True},
                {"name": "bool_field", "type": "boolean", "required": True},
            ]
        }

        Model = schema_to_pydantic("TypeModel", schema)

        instance = Model(
            text_field="hello",
            number_field=3.14,
            int_field=42,
            bool_field=True,
        )

        assert instance.text_field == "hello"
        assert instance.number_field == 3.14
        assert instance.int_field == 42
        assert instance.bool_field is True

    def test_field_descriptions(self):
        """Test field descriptions are preserved."""
        schema = {
            "fields": [
                {
                    "name": "amount",
                    "type": "currency",
                    "required": True,
                    "description": "Total amount in USD",
                }
            ]
        }

        Model = schema_to_pydantic("DescModel", schema)
        field_info = Model.model_fields["amount"]

        assert "Total amount" in field_info.description

    def test_empty_schema(self):
        """Test empty schema creates empty model."""
        schema = {"fields": []}
        Model = schema_to_pydantic("EmptyModel", schema)
        instance = Model()
        assert instance is not None


@pytest.mark.unit
class TestGenerateExtractionModels:
    """Tests for preset model generation."""

    def test_generates_preset_models(self):
        """Test that preset models are generated."""
        models = generate_extraction_models()

        assert "invoice" in models
        assert "receipt" in models
        assert "id_card" in models
        assert "business_card" in models

    def test_model_names_are_pascal_case(self):
        """Test model names are converted to PascalCase."""
        models = generate_extraction_models()

        for name, model in models.items():
            assert model.__name__[0].isupper()


@pytest.mark.unit
class TestResponseModels:
    """Tests for pre-built response models."""

    def test_extraction_result(self):
        """Test ExtractionResult model."""
        result = ExtractionResult(
            success=True,
            text="Extracted text",
            data={"field": "value"},
            confidence=0.95,
        )

        assert result.success is True
        assert result.text == "Extracted text"
        assert result.confidence == 0.95

    def test_batch_job_status(self):
        """Test BatchJobStatus model."""
        from datetime import datetime

        status = BatchJobStatus(
            job_id="job-123",
            status="processing",
            total_items=10,
            processed_items=5,
            failed_items=1,
            created_at=datetime.utcnow(),
        )

        assert status.job_id == "job-123"
        assert status.total_items == 10
        assert status.processed_items == 5
