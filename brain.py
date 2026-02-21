# requires: google-genai, python-dotenv, pydantic
"""
brain.py — Panacea Reasoning Engine
Fully dynamic AI backend — no hardcoded procedures.
Uses the google-genai SDK with Gemini thinking mode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ──────────────────────────────────────────────
# Gemini client (google-genai SDK)
# ──────────────────────────────────────────────

_client = None 

MODEL = "gemini-2.5-flash"


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return None

    from google import genai

    _client = genai.Client(api_key=api_key)
    return _client


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────

class SyllabusStep(BaseModel):
    step_name: str
    target_tool_key: str
    instruction: str
    medical_rationale: str


class TrainingSyllabus(BaseModel):
    steps: list[SyllabusStep]


class SyllabusError(BaseModel):
    error: str


# ──────────────────────────────────────────────
# Action result types
# ──────────────────────────────────────────────

@dataclass
class ActionSuccess:
    status: str = "success"
    tool: str = ""
    message: str = ""


@dataclass
class ActionCorrection:
    status: str = "correction"
    wrong_tool: str = ""
    target_tool: str = ""
    message: str = ""


# ──────────────────────────────────────────────
# Standard YOLO tool keys
# ──────────────────────────────────────────────
# Gemini is instructed to use these when applicable.
# Also used to generate distractor tools for the
# video-callback simulation.

STANDARD_TOOL_KEYS = [
    "scalpel", "forceps", "scissors", "retractor", "needle_driver",
    "suture", "clamp", "trocar", "stapler", "cautery",
    "probe", "curette", "elevator", "speculum", "dilator",
    "catheter", "syringe", "gauze", "gloves", "drape",
    "suction", "irrigator", "grasper", "dissector", "clip_applier",
    "bone_saw", "rongeur", "periosteal_elevator", "chisel",
    "phaco_handpiece", "iol_injector", "capsulorhexis_forceps",
    "cannula", "dermatome", "bipolar_forceps", "laryngoscope",
    "endoscope", "aspirator", "hemostat", "tourniquet",
]


# ──────────────────────────────────────────────
# 1. Dynamic syllabus generation
# ──────────────────────────────────────────────

_SYLLABUS_SYSTEM = """\
You are a Surgical Professor at Webster University. Your responses are \
grounded in WHO Surgical Safety protocols and ACS (American College of \
Surgeons) Instrument Standards.

TASK
Generate a 3-to-5 step training syllabus for the medical/surgical procedure \
the user describes.

GUARDRAIL
If the user input is NOT a recognizable medical or surgical procedure \
(e.g. cooking, programming, sports, casual questions), return ONLY:
{"error": "Please enter a valid medical or surgical procedure for training."}

TOOL KEY RULES
The `target_tool_key` field MUST be a standardized, lowercase, snake_case \
name that can serve as a YOLO object-detection class label. \
Use these canonical keys whenever the instrument matches:
scalpel, forceps, scissors, retractor, needle_driver, suture, clamp, \
trocar, stapler, cautery, probe, curette, elevator, speculum, dilator, \
catheter, syringe, gauze, gloves, drape, suction, irrigator, grasper, \
dissector, clip_applier, bone_saw, rongeur, periosteal_elevator, chisel, \
phaco_handpiece, iol_injector, capsulorhexis_forceps, cannula, dermatome, \
bipolar_forceps, laryngoscope, endoscope, aspirator, hemostat, tourniquet.
If no standard key fits, create a clear snake_case key (e.g. "corneal_shield").

RESPONSE FORMAT — valid JSON only:
{"steps": [
  {"step_name": "...", "target_tool_key": "...", \
"instruction": "...", "medical_rationale": "..."},
  ...
]}
"""


def generate_dynamic_syllabus(user_input: str) -> TrainingSyllabus | SyllabusError:
    """Send any procedure name to Gemini and get a validated syllabus or error."""
    client = _get_client()

    if not client:
        return SyllabusError(
            error="No API key configured. Add GOOGLE_API_KEY to your .env file.",
        )

    try:
        from google.genai import types

        response = client.models.generate_content(
            model=MODEL,
            contents=f"Generate a training syllabus for: {user_input}",
            config=types.GenerateContentConfig(
                system_instruction=_SYLLABUS_SYSTEM,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
            ),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
    except Exception as exc:
        return SyllabusError(error=f"Gemini API error: {exc}")

    if "error" in data:
        return SyllabusError(error=data["error"])

    try:
        return TrainingSyllabus(**data)
    except Exception as exc:
        return SyllabusError(error=f"Schema validation failed: {exc}")


# ──────────────────────────────────────────────
# 2. Proctoring engine
# ──────────────────────────────────────────────

_COACHING_SYSTEM = """\
You are a Surgical Proctor grounded in WHO Surgical Safety protocols. \
The student made an instrument error during training. \
In ONE sentence, explain the clinical danger or inefficiency of their \
choice. Be direct, evidence-based, and educational."""

_COACHING_PROMPT = (
    'Current step: "{instruction}"\n'
    "Required tool: {target_tool}\n"
    "Student picked up: {wrong_tool}\n\n"
    "Why is this wrong?"
)


def check_student_action(
    detected_tools: list[str],
    current_target_tool: str,
    current_instruction: str = "",
    current_rationale: str = "",
) -> ActionSuccess | ActionCorrection:
    """Compare detections against the target and return success or coaching."""
    if current_target_tool in detected_tools:
        return ActionSuccess(
            tool=current_target_tool,
            message=current_rationale or f"{current_target_tool} correctly identified.",
        )

    wrong = [t for t in detected_tools if t != current_target_tool]
    if not wrong:
        return ActionCorrection(
            wrong_tool="(nothing)",
            target_tool=current_target_tool,
            message=f"No valid tool detected. Present {current_target_tool} to the camera.",
        )

    wrong_tool = wrong[0]
    coaching = _get_coaching_tip(current_instruction, current_target_tool, wrong_tool)

    return ActionCorrection(
        wrong_tool=wrong_tool,
        target_tool=current_target_tool,
        message=coaching,
    )


def _get_coaching_tip(instruction: str, target_tool: str, wrong_tool: str) -> str:
    """Generate a coaching tip via Gemini with thinking mode enabled."""
    client = _get_client()

    if client:
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=MODEL,
                contents=_COACHING_PROMPT.format(
                    instruction=instruction,
                    target_tool=target_tool,
                    wrong_tool=wrong_tool,
                ),
                config=types.GenerateContentConfig(
                    system_instruction=_COACHING_SYSTEM,
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                ),
            )
            return response.text.strip()
        except Exception:
            pass

    return (
        f"'{wrong_tool}' is not appropriate at this step — you need "
        f"'{target_tool}' to safely proceed with: {instruction}."
    )
