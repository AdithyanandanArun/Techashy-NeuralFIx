import logging
from pathlib import Path
from app.services.groq_service import analyze_image_with_groq

logger = logging.getLogger(__name__)

async def analyze_equipment_image(image_path: str) -> dict:
    try:
        filename = Path(image_path).name
        analysis = await analyze_image_with_groq(filename)
        return {"success": True, "analysis": analysis, "image_path": image_path}
    except Exception as e:
        logger.error(f"Vision error: {e}")
        return {"success": False, "error": str(e), "analysis": None}
