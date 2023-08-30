from fastapi import APIRouter, HTTPException

from app.services.synonym_handler import SynonymHandler
from app.models.response import AugmentationResponse
from app.models.input_body import SynonymBody

router = APIRouter()
synonym_service = SynonymHandler()


@router.post("/", response_model=AugmentationResponse)
async def synonym_augmentation(request: SynonymBody):
    """
    Augmentation by inserting/substituting token with its synonyms.
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = synonym_service.augment(action=request.action_str,
                                                 text=request.text_str,
                                                 p_aug=request.p_aug_str,
                                                 min_aug=request.min_aug_str,
                                                 max_aug=request.max_aug_str,
                                                 num_similar=request.num_similar_str,
                                                 num_keep=request.num_keep_str,
                                                 exclude=request.exclude_lst,
                                                 is_segmented=request.is_segmented_bool,
                                                 segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)
