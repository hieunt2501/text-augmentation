from fastapi import APIRouter, HTTPException

from app.services.backtranslation_handler import BackTranslationHandler
from app.models.response import AugmentationResponse
from app.models.input_body import BackTranslationBody

router = APIRouter()
back_translation_service = BackTranslationHandler()


@router.post("/", response_model=AugmentationResponse)
async def back_translation(request: BackTranslationBody):
    """
    Augmentation by back-translating sentence.\n
    E.g. Vi -> En -> Vi
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = back_translation_service.augment(text=request.text_str,
                                                          src_language=request.src_language_str,
                                                          languages=request.languages_lst,
                                                          exclude=request.exclude_lst,
                                                          is_segmented=request.is_segmented_bool,
                                                          segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)
