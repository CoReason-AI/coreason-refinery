import sys
from enum import Enum
from typing import Any, Dict
from unittest.mock import MagicMock

from pydantic import BaseModel

# Mock coreason_validator.schemas.knowledge if it's missing in the environment
# This allows tests to run even if the installed package is outdated or missing the module
try:
    import coreason_validator.schemas.knowledge  # noqa: F401
except ImportError:

    class ArtifactType(str, Enum):
        TEXT = "TEXT"

    class KnowledgeArtifact(BaseModel):
        content: str
        source_location: Dict[str, Any]
        source_urn: str
        artifact_type: ArtifactType = ArtifactType.TEXT

    mock_module = MagicMock()
    mock_module.ArtifactType = ArtifactType
    mock_module.KnowledgeArtifact = KnowledgeArtifact

    sys.modules["coreason_validator.schemas.knowledge"] = mock_module
