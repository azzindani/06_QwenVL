"""Contract analysis task handler."""

from typing import Any, Dict, List, Optional, Union

from PIL import Image

from ..utils.parsers import parse_json_from_markdown
from .base import BaseTaskHandler, TaskResult, TaskType, register_handler


@register_handler(TaskType.CONTRACT)
class ContractHandler(BaseTaskHandler):
    """Handler for contract analysis tasks."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.CONTRACT

    @property
    def system_prompt(self) -> str:
        return (
            "You are a helpful assistant specialized in contract analysis. "
            "Extract key information from contracts including parties involved, "
            "dates, terms, clauses, obligations, and important provisions."
        )

    def process(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        extract_clauses: bool = True,
        extract_obligations: bool = True,
        **kwargs,
    ) -> TaskResult:
        """
        Analyze a contract document.

        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompt
            extract_clauses: Whether to extract individual clauses
            extract_obligations: Whether to extract obligations

        Returns:
            TaskResult with contract analysis
        """
        img = self._load_image(image)

        user_prompt = prompt or self._build_contract_prompt(
            extract_clauses, extract_obligations
        )

        messages = self._build_messages(img, user_prompt)
        response = self._generate(messages, **kwargs)

        # Parse contract data
        data = parse_json_from_markdown(response)

        if data:
            parties = data.get("parties", [])
            dates = data.get("dates", {})
            clauses = data.get("clauses", [])
            obligations = data.get("obligations", [])
            key_terms = data.get("key_terms", {})
        else:
            parties = []
            dates = {}
            clauses = []
            obligations = []
            key_terms = {}

        return TaskResult(
            text=response,
            data={
                "parties": parties,
                "dates": dates,
                "clauses": clauses,
                "obligations": obligations,
                "key_terms": key_terms,
            },
            metadata={
                "party_count": len(parties),
                "clause_count": len(clauses),
                "obligation_count": len(obligations),
            },
        )

    def _build_contract_prompt(
        self, extract_clauses: bool, extract_obligations: bool
    ) -> str:
        """Build contract analysis prompt."""
        prompt_parts = [
            "Analyze this contract document and extract:\n\n",
            "1. **Parties**: All parties involved with their roles\n",
            "2. **Dates**: Effective date, termination date, and other key dates\n",
        ]

        if extract_clauses:
            prompt_parts.append(
                "3. **Clauses**: Key clauses with their titles and summaries\n"
            )

        if extract_obligations:
            prompt_parts.append(
                "4. **Obligations**: What each party must do\n"
            )

        prompt_parts.append(
            "5. **Key Terms**: Important terms like payment, duration, termination conditions\n\n"
            "Return as JSON:\n"
            "```json\n"
            "{\n"
            '  "parties": [\n'
            '    {\n'
            '      "name": "Party Name",\n'
            '      "role": "Buyer/Seller/Service Provider/Client",\n'
            '      "address": "Address if present",\n'
            '      "representative": "Signing person"\n'
            '    }\n'
            '  ],\n'
            '  "dates": {\n'
            '    "effective_date": "2024-01-15",\n'
            '    "termination_date": "2025-01-15",\n'
            '    "signing_date": "2024-01-10"\n'
            '  },\n'
            '  "clauses": [\n'
            '    {\n'
            '      "number": "1",\n'
            '      "title": "Clause Title",\n'
            '      "summary": "Brief summary of clause content",\n'
            '      "type": "standard|custom|boilerplate"\n'
            '    }\n'
            '  ],\n'
            '  "obligations": [\n'
            '    {\n'
            '      "party": "Party name",\n'
            '      "obligation": "What they must do",\n'
            '      "deadline": "Date or condition",\n'
            '      "consequence": "What happens if not fulfilled"\n'
            '    }\n'
            '  ],\n'
            '  "key_terms": {\n'
            '    "contract_value": "Total value",\n'
            '    "payment_terms": "Payment schedule",\n'
            '    "duration": "Contract duration",\n'
            '    "termination_clause": "How to terminate",\n'
            '    "governing_law": "Jurisdiction",\n'
            '    "dispute_resolution": "Arbitration/Court"\n'
            '  }\n'
            '}\n'
            "```"
        )

        return "".join(prompt_parts)

    def extract_parties(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Extract only party information from contract.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with party information
        """
        prompt = (
            "Extract only the parties involved in this contract.\n\n"
            "Return as JSON:\n"
            "```json\n"
            "{\n"
            '  "parties": [\n'
            '    {\n'
            '      "name": "Party Name",\n'
            '      "role": "Role in contract",\n'
            '      "address": "Address",\n'
            '      "representative": "Signing person"\n'
            '    }\n'
            '  ]\n'
            '}\n'
            "```"
        )
        return self.process(image, prompt=prompt, **kwargs)

    def extract_key_dates(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> TaskResult:
        """
        Extract only key dates from contract.

        Args:
            image: Image path or PIL Image

        Returns:
            TaskResult with date information
        """
        prompt = (
            "Extract all important dates from this contract.\n\n"
            "Return as JSON:\n"
            "```json\n"
            "{\n"
            '  "dates": {\n'
            '    "effective_date": "date",\n'
            '    "termination_date": "date",\n'
            '    "signing_date": "date",\n'
            '    "other_dates": [{"name": "date name", "date": "date value"}]\n'
            '  }\n'
            '}\n'
            "```"
        )
        return self.process(image, prompt=prompt, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("CONTRACT HANDLER TEST")
    print("=" * 60)
    print("  Contract handler registered successfully")
    print(f"  Task type: {TaskType.CONTRACT}")
    print("=" * 60)
