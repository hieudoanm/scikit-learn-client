"""Algorithm catalogue endpoint."""

from fastapi import APIRouter, Query

from src.api.schemas import (
    AlgorithmResponse,
    AlgorithmsListResponse,
    ArtefactFormat,
    TaskType,
)
from src.training.catalogue import ALGORITHMS

router = APIRouter(tags=['algorithms'])


@router.get(
    '/api/v1/algorithms', summary='Algorithm catalogue', response_model=AlgorithmsListResponse
)
async def list_algorithms(
    task_type: TaskType | None = Query(None, alias='taskType'),
) -> AlgorithmsListResponse:
    """Return the allow-listed algorithm catalogue, optionally filtered by task type."""
    algorithms = []
    for algo_id, spec in ALGORITHMS.items():
        if task_type is not None and spec['task_type'] != task_type.value:
            continue
        algorithms.append(
            AlgorithmResponse(
                algorithmId=algo_id,
                name=spec['name'],
                taskType=TaskType(spec['task_type']),
                family=spec['family'],
                description=spec['description'],
                defaultHyperparameters=spec['default_hyperparameters'],
                exports=[ArtefactFormat(e) for e in spec['exports']],
            )
        )
    return AlgorithmsListResponse(algorithms=algorithms)
