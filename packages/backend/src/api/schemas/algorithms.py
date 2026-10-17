"""Algorithm catalogue schemas."""

from typing import Any

from pydantic import BaseModel, Field

from src.api.schemas.shared import ArtefactFormat, TaskType


class AlgorithmResponse(BaseModel):
    algorithmId: str
    name: str
    taskType: TaskType
    family: str = 'sklearn'
    description: str
    defaultHyperparameters: dict[str, Any] = Field(default_factory=dict)
    exports: list[ArtefactFormat] = Field(default_factory=list)


class AlgorithmsListResponse(BaseModel):
    algorithms: list[AlgorithmResponse]
