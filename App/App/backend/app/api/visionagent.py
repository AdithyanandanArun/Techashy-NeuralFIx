import ollama
import base64, json
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from groq import Groq
from app.core.config import get_settings

router = APIRouter(prefix="/vision", tags=["Vision Agent"])
settings = get_settings()
groq_client = Groq(api_key=settings.groq_api_key)

PROMPT = """You are a network equipment technician.
Analyse this image carefully and return ONLY valid JSON, nothing else:
{
  "device_type": "router | switch | modem | access_point | unknown",
  "brand_model": "brand and model string, or null if unclear",
  "led_states": [
    {"label": "WAN", "color": "red | green | amber | off | blinking", "blinking": false}
  ],
  "unplugged_ports": ["port 1", "port 3"],
  "visible_damage": "description of damage, or null",
  "overall_assessment": "one sentence summary of the problem",
  "confidence": 0.85
}
Return ONLY the JSON object. No explanation, no markdown fences."""

class ChatMessage(BaseModel):
    role: str
    content: str

class VisionChatRequest(BaseModel):
    messages: List[ChatMessage]
    vision_context: Dict[str, Any]

def analyse_image_bytes(image_bytes: bytes) -> dict:
    # encode image to base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # call local LLaVA via Ollama
    response = ollama.generate(
        model="llava-llama3",        # or "llava"
        prompt=PROMPT,
        images=[img_b64],   # list of base64 strings
        stream=False
    )
    
    raw = response["response"].strip()
    
    # strip markdown fences if model adds them anyway
    raw = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        parsed_json = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: model didn't return clean JSON
        parsed_json = {
            "device_type": "unknown",
            "overall_assessment": raw,  # use raw text as fallback
            "confidence": 0.0,
            "error": "JSON parse failed"
        }

    # ─── PART 2: Ask Groq for detailed fix steps based on the vision JSON
    summary_prompt = f"""You are an expert network technician. 
A physical inspection of the user's network equipment yielded the following findings:
{json.dumps(parsed_json, indent=2)}

Please provide a highly detailed, step-by-step markdown response explaining exactly how the user can troubleshoot and fix the issues identified. 
Structure it clearly with headings, bullet points, and any relevant warnings.
"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful and expert network support assistant."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        parsed_json["detailed_fix_summary"] = completion.choices[0].message.content
    except Exception as e:
        parsed_json["detailed_fix_summary"] = f"Failed to generate fix summary: {str(e)}"

    return parsed_json

@router.post("/analyse")
async def analyse_equipment_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    result = analyse_image_bytes(contents)
    return result

@router.post("/chat")
async def vision_chat(payload: VisionChatRequest):
    """
    Follow-up chat about a specific hardware image that was just analyzed.
    Uses Groq to answer user questions with the vision context in mind.
    """
    
    # System prompt provides context about the equipment
    system_msg = f"""You are an expert network technician assisting a user with their equipment.
Earlier, an AI vision model analyzed a photo of their equipment and found:
{json.dumps(payload.vision_context, indent=2)}

Answer the user's follow-up questions about this equipment based ONLY on this context and your general networking knowledge. Be concise and helpful. Wait for them to ask.
"""
    
    api_messages = [{"role": "system", "content": system_msg}]
    for m in payload.messages:
        api_messages.append({"role": m.role, "content": m.content})
        
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=api_messages,
            temperature=0.5,
            max_tokens=512,
            top_p=1,
            stream=False,
        )
        return {"reply": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")
