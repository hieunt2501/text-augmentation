from fastapi import APIRouter, HTTPException
from loguru import logger

from app.services.pipeline_handler import PipelineHandler
from app.models.response import AugmentationResponse
from app.models.input_body import PipelineBody

router = APIRouter()
pipeline_service = PipelineHandler()
fmt = "{time} - {name} - {level} - {message}"
logger.add("logs/pipeline_api.log", level="DEBUG", format=fmt, backtrace=True)


@router.post("/", response_model=AugmentationResponse)
async def pipeline_augmentation(request: PipelineBody):
    """
    Augmentation with specific pipeline
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = pipeline_service.augment(text=request.text_str,
                                                  exclude=request.exclude_lst,
                                                  pipeline=request.pipeline_lst,
                                                  is_segmented=request.is_segmented_bool,
                                                  segment=request.segment_bool,
                                                  n_sent=request.n_sent_str)
    except Exception as e:
        logger.error(f"Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)
