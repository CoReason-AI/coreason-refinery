import re


def _clean_header_text(text: str) -> str:
    """Strip leading/trailing Markdown hashes and whitespace for context usage."""
    # 1. Strip leading hashes
    text = re.sub(r"^\s*#+\s*", "", text)
    # 2. Strip trailing hashes
    text = re.sub(r"\s*#+\s*$", "", text)
    return text.strip()


input_text = "## Clean Header ##"
output = _clean_header_text(input_text)
print(f"Input: '{input_text}'")
print(f"Output: '{output}'")
expected = "Clean Header"
assert output == expected, f"Expected '{expected}', got '{output}'"
print("Assertion Passed")
