from fastapi import APIRouter

from app.api.routes import synonym, back_translation, blank, spelling, dependency_tree, pipeline

router = APIRouter()
router.include_router(synonym.router, tags=["synonym"], prefix="/synonym")
router.include_router(back_translation.router, tags=["backtranslation"], prefix="/backtranslation")
router.include_router(blank.router, tags=["blank_noise"], prefix="/blank")
router.include_router(spelling.router, tags=["spelling"], prefix="/spelling")
router.include_router(dependency_tree.router, tags=["dependency_tree"], prefix="/dependency_tree")
router.include_router(pipeline.router, tags=["pipeline"], prefix="/pipeline")
