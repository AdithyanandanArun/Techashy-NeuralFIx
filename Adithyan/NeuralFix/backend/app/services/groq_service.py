"""
NeuralFix — Groq AI Service
Uses llama-3.1-8b-instant for fast IT troubleshooting across all categories.
"""
import logging
from typing import List
from groq import Groq
from app.core.config import get_settings
from app.services.rag_service import retrieve_context

logger = logging.getLogger(__name__)
settings = get_settings()
client = Groq(api_key=settings.groq_api_key)
MODEL = "llama-3.1-8b-instant"

BASE_SYSTEM_PROMPT = """You are NeuralFix, a friendly AI assistant that helps non-technical people fix technology problems. You work in remote offices, schools, clinics, and field locations where no IT person is available.

You can help with ALL of these:
- 🌐 Networking & WiFi — no internet, slow connection, router issues
- 💻 Computers & Laptops — slow, frozen, won't start, blue screen
- 🖨️ Printers & Scanners — not printing, paper jams, not detected
- 📱 Mobile Phones & Tablets — won't charge, apps crashing, storage full
- 💿 Software & Apps — won't open, error messages, updates failing
- 📺 TVs, Projectors & Displays — no signal, wrong resolution, HDMI issues
- 🔑 Passwords & Accounts — locked out, forgot password, account issues
- 🏠 Smart Devices & IoT — smart lights, speakers, thermostats not responding

Your rules:
- Use SIMPLE language — no jargon, no technical terms unless explained
- Ask ONE question at a time to narrow down the problem
- Give SHORT numbered steps — one action per step
- Always confirm a step worked before moving on
- Use emoji for visual cues: ✅ done, ⚠️ caution, 🔴 error, 💡 tip, 🔄 restart
- Be encouraging — users are frustrated, be patient and calm
- Start with the SIMPLEST fix first (restart, check cables, check power)
- After 4-5 failed steps, offer to generate a diagnostic report for IT support

First response rule: Always start by asking ONE clarifying question to understand the exact problem before giving steps.

Common first fixes to always try:
1. Restart/reboot the device
2. Check all cable connections
3. Check if it's powered on
4. Check if others have the same problem (rules out user error)"""


def build_system_prompt(rag_context: str = "", vision_context: str = "") -> str:
    prompt = BASE_SYSTEM_PROMPT
    if vision_context:
        prompt += f"\n\n[DEVICE IMAGE ANALYSIS]\n{vision_context}"
    if rag_context:
        prompt += f"\n\n[RELEVANT DOCUMENTATION]\n{rag_context}\nUse this to inform your guidance but keep language simple."
    return prompt


async def get_chat_response(messages: List[dict], latest_user_message: str, vision_context: str = "") -> str:
    rag_context = retrieve_context(latest_user_message, k=3)
    system = build_system_prompt(rag_context=rag_context, vision_context=vision_context)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages if m["role"] in ("user", "assistant")
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + api_messages,
        max_tokens=1024,
        temperature=0.4,
    )
    return response.choices[0].message.content


async def analyze_image_with_groq(image_description: str) -> str:
    prompt = f"""A user has uploaded an image of a device or tech equipment: "{image_description}"

Provide a brief analysis:
1. What type of device this likely is
2. What visible clues might indicate the problem (lights, screen, cables)
3. What to look for to troubleshoot it

Keep it simple and practical for a non-technical user."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    return response.choices[0].message.content


async def generate_diagnostic_report(session_data: dict) -> str:
    messages_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'NeuralFix'}: {m['content']}"
        for m in session_data.get("messages", [])
    ])
    prompt = f"""Generate a structured IT diagnostic report from this troubleshooting session.

App: NeuralFix
Session: {session_data.get('title', 'Tech Issue')}
Category: {session_data.get('category', 'general')}

Conversation:
{messages_text}

Write the report with these exact section headers using **SECTION NAME** format:

**INCIDENT SUMMARY**
**DEVICE / SYSTEM INFORMATION**
**REPORTED SYMPTOMS**
**TROUBLESHOOTING STEPS ATTEMPTED**
**CURRENT STATUS**
**ROOT CAUSE ANALYSIS**
**RECOMMENDED NEXT STEPS**
**PRIORITY LEVEL**

Be factual and concise. This will be read by IT support technicians."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
    )
    return response.choices[0].message.content
