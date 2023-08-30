from fastapi import APIRouter, HTTPException

from app.services.blank_noise_handler import BlankNoiseHandler
from app.models.response import AugmentationResponse
from app.models.input_body import BlankNoiseBody

router = APIRouter()
blank_noise_service = BlankNoiseHandler()


@router.post("/", response_model=AugmentationResponse)
async def blank_augmentation(request: BlankNoiseBody):
    """
    Augmentation by replacing token with _ (blank) token. \n
    Ref: https://arxiv.org/pdf/1703.02573.pdf
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = blank_noise_service.augment(action=None,
                                                     text=request.text_str,
                                                     p_aug=request.p_aug_str,
                                                     min_aug=request.min_aug_str,
                                                     max_aug=request.max_aug_str,
                                                     exclude=request.exclude_lst,
                                                     is_segmented=request.is_segmented_bool,
                                                     segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)
