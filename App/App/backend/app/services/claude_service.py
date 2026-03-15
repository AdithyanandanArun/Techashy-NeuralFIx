"""
Claude Chat Service
===================
Handles all LLM interactions. Injects RAG context and vision analysis
into prompts before sending to Claude.
"""

import logging
from typing import List, Optional
import anthropic

from app.core.config import get_settings
from app.services.rag_service import retrieve_context

logger = logging.getLogger(__name__)
settings = get_settings()

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

BASE_SYSTEM_PROMPT = """You are NetFixAI, an expert network troubleshooting assistant for non-technical staff in remote offices and field locations.

Your mission: Guide users to resolve network connectivity problems using simple, jargon-free language.

Behavior:
- Ask one clarifying question at a time to narrow down the problem
- Use numbered steps for instructions — keep each step short and actionable
- Confirm each step completed before moving forward
- Use emoji sparingly: ✅ success, ⚠️ caution, 🔴 error, 💡 tip
- Never assume the user has technical knowledge
- Track what has been tried in the conversation

Troubleshooting approach (follow OSI layers bottom-up):
1. Physical: power, cables, LEDs, physical damage
2. Link: device restart, factory reset
3. Network: IP settings, DHCP, DNS
4. Application: browser, specific service issues

LED interpretation guide:
- Solid green = normal/connected
- Blinking green = activity (normal)
- Solid/blinking amber or orange = warning or low-speed link
- Red or blinking red = error or no link
- No light = no power or port disabled

When to escalate:
- After 4-5 troubleshooting steps without resolution
- When physical damage is visible
- When the issue affects multiple devices
- Always offer to generate a Diagnostic Report for IT support

Keep responses concise — users are often stressed and in a hurry."""


def build_system_prompt(rag_context: str = "", vision_context: str = "") -> str:
    """Builds the full system prompt by appending RAG and vision context."""
    prompt = BASE_SYSTEM_PROMPT

    if vision_context:
        prompt += f"\n\n{'='*50}\nEQUIPMENT IMAGE ANALYSIS (from uploaded photo):\n{vision_context}\n{'='*50}"

    if rag_context:
        prompt += f"\n\n{'='*50}\nRELEVANT DOCUMENTATION (from knowledge base):\n{rag_context}\n{'='*50}\nUse the above documentation to inform your guidance, but translate technical content into simple language."

    return prompt


async def get_chat_response(
    messages: List[dict],
    latest_user_message: str,
    vision_context: str = "",
) -> str:
    """
    Main chat function. Retrieves RAG context, builds prompt, calls Claude.

    Args:
        messages: Full conversation history [{role, content}]
        latest_user_message: The new user message (used for RAG retrieval)
        vision_context: Optional formatted vision analysis string

    Returns:
        Claude's response text
    """
    # 1. Retrieve RAG context based on the latest message
    rag_context = retrieve_context(latest_user_message, k=4)

    # 2. Build system prompt
    system = build_system_prompt(rag_context=rag_context, vision_context=vision_context)

    # 3. Format messages for Anthropic API
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]

    # 4. Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=api_messages,
    )

    return response.content[0].text


async def generate_diagnostic_report(session_data: dict) -> str:
    """
    Generates a structured diagnostic report from a session.
    Used by the /api/reports/generate endpoint.
    """
    messages_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'NetFixAI'}: {m['content']}"
        for m in session_data.get("messages", [])
    ])

    vision_summary = ""
    if session_data.get("device_info"):
        vision_summary = f"\nEquipment Analysis: {session_data['device_info']}\n"

    prompt = f"""Generate a structured IT diagnostic report from this troubleshooting session.

Session Title: {session_data.get('title', 'Network Issue')}
{vision_summary}
Conversation:
{messages_text}

Write the report with these exact sections using **SECTION NAME** as headers:

**INCIDENT SUMMARY**
Brief 2-sentence description of the reported problem.

**DEVICE INFORMATION**
Equipment type, brand/model if known, location if mentioned, physical identifiers.

**REPORTED SYMPTOMS**
Bullet list of specific symptoms the user described.

**TROUBLESHOOTING STEPS ATTEMPTED**
Numbered list of every step tried and the result of each.

**CURRENT STATUS**
One of: Resolved / Partially Resolved / Unresolved — with one sentence explanation.

**ROOT CAUSE ANALYSIS**
Most likely cause(s) based on available evidence.

**RECOMMENDED NEXT STEPS**
What the IT support technician should do when they arrive or call in.

**PRIORITY LEVEL**
Critical / High / Medium / Low — with one sentence justification.

Be factual and concise. This report will be read by IT support technicians."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
