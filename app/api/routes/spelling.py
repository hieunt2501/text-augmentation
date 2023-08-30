from fastapi import APIRouter, HTTPException

from app.models.response import AugmentationResponse
from app.models.input_body import TypoBody, AccentBody, SpellingReplaceBody, WordAugmentationBody, CharAugmentationBody
from app.services.spelling_handler import (
    TypoHandler,
    AccentHandler,
    SpellingReplacementHandler,
    WordHandler,
    CharHandler
)

router = APIRouter()
typo_aug = TypoHandler()
accent_aug = AccentHandler()
spell_aug = SpellingReplacementHandler()
word_aug = WordHandler()
char_aug = CharHandler()


@router.post("/typo", response_model=AugmentationResponse)
async def typo_augmentation(request: TypoBody):
    """
    Augmentation by replacing token with its telex/vni/keyboard error.
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = typo_aug.augment(action=request.action_str,
                                          text=request.text_str,
                                          p_aug=request.p_aug_str,
                                          min_aug=request.min_aug_str,
                                          max_aug=request.max_aug_str,
                                          p_char_aug=request.p_aug_str,
                                          exclude=request.exclude_lst,
                                          is_segmented=request.is_segmented_bool,
                                          segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)


@router.post("/accent", response_model=AugmentationResponse)
async def accent_augmentation(request: AccentBody):
    """
    Augmentation by replacing token with missing/none/wrong accent.
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = accent_aug.augment(action=request.action_str,
                                            text=request.text_str,
                                            p_aug=request.p_aug_str,
                                            min_aug=request.min_aug_str,
                                            max_aug=request.max_aug_str,
                                            p_char_aug=request.p_aug_str,
                                            exclude=request.exclude_lst,
                                            is_segmented=request.is_segmented_bool,
                                            segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)


@router.post("/char", response_model=AugmentationResponse)
async def char_augmentation(request: CharAugmentationBody):
    """
    Augmentation at character level.
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = char_aug.augment(action=request.action_str,
                                          text=request.text_str,
                                          p_aug=request.p_aug_str,
                                          min_aug=request.min_aug_str,
                                          max_aug=request.max_aug_str,
                                          p_char_aug=request.p_aug_str,
                                          exclude=request.exclude_lst,
                                          is_segmented=request.is_segmented_bool,
                                          segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)


@router.post("/word", response_model=AugmentationResponse)
async def word_augmentation(request: WordAugmentationBody):
    """
    Augmentation at word level
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = word_aug.augment(action=request.action_str,
                                          text=request.text_str,
                                          p_aug=request.p_aug_str,
                                          min_aug=request.min_aug_str,
                                          max_aug=request.max_aug_str,
                                          p_char_aug=request.p_aug_str,
                                          exclude=request.exclude_lst,
                                          is_segmented=request.is_segmented_bool,
                                          segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)


@router.post("/spelling_replace", response_model=AugmentationResponse)
async def spelling_replace_augmentation(request: SpellingReplaceBody):
    """
    Augmentation by substituting common Vietnamese confuse begin/end pairs.\n
    E.g.
    - x -> s
    - d -> đ, gi, r, v
    - c -> t
    - n --> nh, ng
    """
    if not request:
        raise HTTPException(status_code=404, detail=f"'request' argument invalid!")
    try:
        augmented_text = spell_aug.augment(action=request.action_str,
                                           text=request.text_str,
                                           p_aug=request.p_aug_str,
                                           min_aug=request.min_aug_str,
                                           max_aug=request.max_aug_str,
                                           p_char_aug=request.p_aug_str,
                                           exclude=request.exclude_lst,
                                           is_segmented=request.is_segmented_bool,
                                           segment=request.segment_bool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

    return AugmentationResponse(text=augmented_text, org_text=request.text_str)
