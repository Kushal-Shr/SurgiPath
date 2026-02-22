# requires: google-genai, python-dotenv, pydantic
"""
brain.py — SurgiPath Reasoning Engine (Clarity AI)
Fully dynamic AI backend — no hardcoded procedures.
Uses the google-genai SDK with Gemini thinking mode and Live API.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# ──────────────────────────────────────────────
# Gemini client (google-genai SDK)
# ──────────────────────────────────────────────

_client = None 

MODEL = "gemini-3-flash-preview"
LIVE_MODEL = "gemini-2.0-flash-exp-image-generation"


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
    time_limit_seconds: int = 60
    critical_safety_tip: str


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
# Multimodal Live API — real-time frame analysis
# ──────────────────────────────────────────────

class LiveProctor:
    """Persistent WebSocket to the Gemini Live API for real-time visual analysis."""

    def __init__(self) -> None:
        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=3)
        self._feedback_queue: queue.Queue[str] = queue.Queue(maxsize=50)
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def active(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def start(self, procedure: str) -> None:
        if self.active:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, args=(procedure,), daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._clear(self._frame_queue)

    def send_frame(self, jpeg_bytes: bytes) -> None:
        if not self._running:
            return
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            pass

    def get_feedback(self) -> str | None:
        try:
            return self._feedback_queue.get_nowait()
        except queue.Empty:
            return None

    def drain_all_feedback(self) -> list[str]:
        out: list[str] = []
        while True:
            try:
                out.append(self._feedback_queue.get_nowait())
            except queue.Empty:
                break
        return out

    # ── internals ──

    @staticmethod
    def _clear(q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _post(self, text: str) -> None:
        if self._feedback_queue.full():
            try:
                self._feedback_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._feedback_queue.put_nowait(text)
        except queue.Full:
            pass

    def _run(self, procedure: str) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._session(procedure))
        except Exception as exc:
            self._post(f"[Clarity] Session ended: {exc}")
        finally:
            loop.close()

    async def _session(self, procedure: str) -> None:
        client = _get_client()
        if not client:
            self._post("[Clarity] No API key — live analysis unavailable.")
            return

        try:
            from google.genai import types
        except ImportError:
            self._post("[Clarity] SDK does not support Live API.")
            return

        sys_instr = (
            "You are Clarity, the real-time AI surgical proctor for "
            "SurgiPath. "
            f"The student is training on: {procedure}. "
            "You receive live video frames from their workspace. "
            "When prompted with OBSERVE, give a 1-2 sentence clinical "
            "observation. Be direct. Do NOT repeat previous observations."
        )

        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=sys_instr,
        )

        try:
            async with client.aio.live.connect(
                model=LIVE_MODEL, config=config,
            ) as session:
                self._post("[Clarity] Live session connected.")
                send_task = asyncio.create_task(self._send_loop(session))
                recv_task = asyncio.create_task(self._recv_loop(session))
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
        except Exception as exc:
            self._post(
                f"[Clarity] Live connection unavailable "
                f"({type(exc).__name__}: {exc}). "
                f"Standard analysis will be used."
            )

    async def _send_loop(self, session) -> None:
        from google.genai import types

        _PROMPT_INTERVAL = 5.0
        _last_prompt = asyncio.get_event_loop().time()
        _frames_sent = 0

        while self._running:
            try:
                data = self._frame_queue.get_nowait()
                await session.send_realtime_input(
                    media=types.Blob(data=data, mime_type="image/jpeg"),
                )
                _frames_sent += 1
            except queue.Empty:
                pass
            except Exception:
                break

            now = asyncio.get_event_loop().time()
            if _frames_sent > 0 and (now - _last_prompt) >= _PROMPT_INTERVAL:
                prompt_text = (
                    "OBSERVE — Give a 1-sentence clinical observation "
                    "about the current frame."
                )
                try:
                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=prompt_text)],
                        ),
                    )
                    _last_prompt = now
                except Exception:
                    pass

            await asyncio.sleep(0.2)

    async def _recv_loop(self, session) -> None:
        _buf: list[str] = []
        try:
            async for response in session.receive():
                if not self._running:
                    break
                sc = getattr(response, "server_content", None)
                if sc is None:
                    continue
                mt = getattr(sc, "model_turn", None)
                if mt:
                    for part in mt.parts:
                        txt = getattr(part, "text", None)
                        if txt:
                            _buf.append(txt)
                turn_complete = getattr(sc, "turn_complete", False)
                if turn_complete and _buf:
                    text = "".join(_buf).strip()
                    _buf.clear()
                    self._post(text)
        except Exception:
            if self._running:
                await asyncio.sleep(1)


_live_proctor_ref: LiveProctor | None = None


def get_live_proctor() -> LiveProctor | None:
    return _live_proctor_ref


def set_live_proctor(proctor: LiveProctor | None) -> None:
    global _live_proctor_ref
    _live_proctor_ref = proctor


# ──────────────────────────────────────────────
# 1. Dynamic syllabus generation
# ──────────────────────────────────────────────

_SYLLABUS_SYSTEM = """\
You are Clarity, the AI proctor for SurgiPath, \
optimized for sub-second surgical assessment. \
Your responses are grounded in WHO Surgical Safety protocols and ACS \
(American College of Surgeons) Instrument Standards.

TASK
Generate a 3-to-5 step training syllabus for the medical/surgical procedure \
the user describes. Each step MUST include a realistic time limit in seconds \
and a critical safety tip.

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

TIME LIMIT RULES
`time_limit_seconds` is the maximum time a trainee should take to \
identify and present the correct tool. Use 30-45s for simple identification \
tasks, 60-90s for moderate tasks, and 90-120s for complex multi-tool steps.

RESPONSE FORMAT — valid JSON only:
{"steps": [
  {"step_name": "...", "target_tool_key": "...", "instruction": "...", \
"time_limit_seconds": 45, "critical_safety_tip": "..."},
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
You are Clarity, the AI proctor for SurgiPath, \
grounded in WHO Surgical Safety protocols. \
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
    current_safety_tip: str = "",
) -> ActionSuccess | ActionCorrection:
    """Compare detections against the target and return success or coaching."""
    if current_target_tool in detected_tools:
        return ActionSuccess(
            tool=current_target_tool,
            message=current_safety_tip or f"{current_target_tool} correctly identified.",
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


# ──────────────────────────────────────────────
# 3. Manual override — skip warning
# ──────────────────────────────────────────────

_SKIP_SYSTEM = """\
You are Clarity, the AI proctor for SurgiPath, \
grounded in WHO Surgical Safety protocols. \
A student is skipping the physical identification of a required instrument. \
In ONE sentence, give a firm but educational warning about the clinical \
importance of verifying this specific tool before proceeding."""

_SKIP_PROMPT = (
    "The student is skipping physical identification of: {target_tool}\n"
    "Current step: \"{instruction}\"\n"
    "Student's reason: {reason}\n\n"
    "Generate a 1-sentence warning."
)

SKIP_PENALTY = 15
TIMER_PENALTY = 20


def generate_skip_warning(
    target_tool: str,
    instruction: str,
    reason: str = "",
) -> str:
    """Call Gemini for a penalty warning when the student skips a tool."""
    client = _get_client()

    if client:
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=MODEL,
                contents=_SKIP_PROMPT.format(
                    target_tool=target_tool,
                    instruction=instruction,
                    reason=reason or "No reason given",
                ),
                config=types.GenerateContentConfig(
                    system_instruction=_SKIP_SYSTEM,
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                ),
            )
            return response.text.strip()
        except Exception:
            pass

    return (
        f"Skipping verification of '{target_tool}' bypasses a critical "
        f"safety checkpoint — in a live procedure this could lead to using "
        f"the wrong instrument, risking patient injury."
    )


# ──────────────────────────────────────────────
# 4. Post-op session report
# ──────────────────────────────────────────────

_REPORT_SYSTEM = """\
You are Clarity, the AI proctor for SurgiPath, \
writing a concise post-training session report. \
Evaluate the student's performance based on the data provided. \
Be direct, clinical, and educational. Use 3-5 bullet points."""

_REPORT_PROMPT = """\
Procedure: {procedure}
Total steps: {total_steps}
Vision-verified steps: {verified}
Manually skipped steps: {skipped}
Final mastery score: {score}/100

Skipped details:
{skip_details}

Write a brief session assessment. If more than 1/3 of steps were skipped, \
label the session as "Clinically Incomplete" and explain why. \
End with one actionable recommendation."""


def generate_session_report(
    procedure: str,
    total_steps: int,
    verified_count: int,
    skipped_steps: list[dict],
    mastery_score: int,
) -> str:
    """Generate a post-op session report via Gemini."""
    skip_details = "\n".join(
        f"  - Step {s.get('step_idx', '?') + 1}: skipped '{s.get('tool', '?')}' "
        f"(reason: {s.get('reason', 'none')})"
        for s in skipped_steps
    ) or "  (none)"

    client = _get_client()

    if client:
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=MODEL,
                contents=_REPORT_PROMPT.format(
                    procedure=procedure,
                    total_steps=total_steps,
                    verified=verified_count,
                    skipped=len(skipped_steps),
                    score=mastery_score,
                    skip_details=skip_details,
                ),
                config=types.GenerateContentConfig(
                    system_instruction=_REPORT_SYSTEM,
                    thinking_config=types.ThinkingConfig(thinking_budget=2048),
                ),
            )
            return response.text.strip()
        except Exception:
            pass

    skipped_count = len(skipped_steps)
    label = "Clinically Incomplete" if skipped_count > total_steps / 3 else "Conditionally Passed"
    return (
        f"**Session Assessment: {label}**\n\n"
        f"- Procedure: {procedure}\n"
        f"- {verified_count}/{total_steps} steps verified by vision\n"
        f"- {skipped_count} step(s) manually skipped (−{skipped_count * SKIP_PENALTY} pts)\n"
        f"- Final mastery score: {mastery_score}/100\n\n"
        f"**Recommendation:** Repeat any skipped steps with physical instruments "
        f"to build reliable muscle memory and tool-recognition confidence."
    )


# ──────────────────────────────────────────────
# 5. Learning resources for the procedure
# ──────────────────────────────────────────────

_RESOURCES_SYSTEM = """\
You are Clarity, the AI proctor for SurgiPath. \
The student just completed a training session. Recommend learning resources \
for this specific procedure so they can deepen their knowledge. \
Return **exactly 5** resources in Markdown bullet-point format. \
Each bullet must include: a descriptive title (bold), the type \
(Textbook Chapter, Video, Journal Article, Online Module, or Atlas), \
and a 1-sentence reason why it is relevant. \
Prioritize open-access or widely available sources when possible."""


def generate_learning_resources(procedure: str) -> str:
    """Ask Gemini for curated learning resources for the given procedure."""
    client = _get_client()
    prompt = (
        f"Procedure: {procedure}\n\n"
        "Recommend 5 high-quality resources (textbooks, surgical videos, "
        "journal articles, online courses, or anatomical atlases) that a "
        "surgical trainee should study to master this procedure."
    )

    if client:
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_RESOURCES_SYSTEM,
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                ),
            )
            return response.text.strip()
        except Exception:
            pass

    return (
        f"- **{procedure} — Surgical Technique Overview** (Textbook Chapter): "
        f"Consult a standard operative surgery textbook for step-by-step guidance.\n"
        f"- **WHO Surgical Safety Checklist** (Online Module): "
        f"Review the WHO checklist to reinforce safety protocols.\n"
        f"- **Anatomical Atlas** (Atlas): "
        f"Study the relevant regional anatomy for this procedure.\n"
        f"- **Peer-reviewed case studies** (Journal Article): "
        f"Search PubMed for recent case reports on {procedure}.\n"
        f"- **Surgical video library** (Video): "
        f"Watch annotated procedure recordings on platforms like WebSurg or GIBLIB."
    )


# ──────────────────────────────────────────────
# 6. Final critique — Clarity live feedback + event log
# ──────────────────────────────────────────────

_CRITIQUE_SYSTEM = """\
You are Clarity, the AI proctor for SurgiPath. \
You are reviewing a completed training session. You have two data sources:

1. **Live Observations** — real-time clinical observations you made \
   while watching the student's video feed.
2. **Event Log** — structured log of tool detections, matches, wrong \
   selections, manual overrides, and timeouts.

Write your assessment as a professional "Surgical Performance Note" in \
Markdown with these sections:

## Clinical Narrative
Reconstruct a chronological story of the session from the observations \
and event log.

## Technique Assessment
Assess tool handling confidence, workspace discipline, and focus.

## Efficiency Analysis
Were tool transitions swift? Highlight prolonged gaps or hesitation.

## Risk Factors
Note any manual overrides, timeouts, or wrong-tool selections and \
their clinical implications.

## Performance Grade
Grade: Excellent / Proficient / Developing / Unsatisfactory.
One actionable recommendation for the next session."""


def generate_final_critique(
    procedure: str,
    clarity_feedback: list[str],
    event_log: list[dict],
    mastery_score: int = 100,
) -> str:
    """Generate a performance note from Clarity's live observations + event log."""
    if not clarity_feedback and not event_log:
        return (
            "\u26a0\ufe0f **No session data recorded.** The session was too "
            "short or the camera was not active.\n\n"
            + _fallback_event_analysis(event_log, mastery_score)
        )

    clarity_text = "\n".join(clarity_feedback) if clarity_feedback else "(none)"

    event_text = "\n".join(
        f"  [{e.get('time', '??:??:??')}] {e.get('type', '?').upper()}: "
        f"{e.get('tool', '')} \u2014 {e.get('detail', '')}"
        for e in event_log
    ) or "  (no events recorded)"

    overrides_count = sum(1 for e in event_log if e.get("type") == "override")

    prompt = (
        f"Procedure: {procedure}\n"
        f"Mastery Score: {mastery_score}/100\n"
        f"Manual overrides: {overrides_count}\n"
        f"Clarity live observations: {len(clarity_feedback)}\n\n"
        f"--- CLARITY LIVE OBSERVATIONS ---\n"
        f"{clarity_text}\n\n"
        f"--- EVENT LOG ---\n"
        f"{event_text}\n\n"
        f"Write the Surgical Performance Note based on this data."
    )

    if overrides_count > 0:
        prompt += (
            f"\n\nCRITICAL: {overrides_count} manual override(s) detected. "
            f"Explain the clinical danger of bypassing instrument verification."
        )

    client = _get_client()
    if not client:
        return _fallback_event_analysis(event_log, mastery_score)

    try:
        from google.genai import types

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_CRITIQUE_SYSTEM,
                thinking_config=types.ThinkingConfig(thinking_budget=4096),
            ),
        )
        return response.text.strip()
    except Exception as exc:
        return (
            f"\u26a0\ufe0f Analysis error: {exc}\n\n"
            + _fallback_event_analysis(event_log, mastery_score)
        )


def _fallback_event_analysis(event_log: list[dict], mastery_score: int) -> str:
    """Event-log-only analysis when Gemini is unavailable."""
    matches = sum(1 for e in event_log if e.get("type") == "detection_match")
    wrongs = sum(1 for e in event_log if e.get("type") == "wrong_tool")
    overrides = sum(1 for e in event_log if e.get("type") == "override")
    total = matches + wrongs + overrides

    override_warning = ""
    if overrides >= 3:
        override_warning = (
            f"\n## \u26a0\ufe0f Manual Override Risk Assessment\n"
            f"**{overrides}** overrides represent a critical failure in "
            f"WHO Surgical Safety protocol.\n"
        )
    elif overrides >= 1:
        override_warning = (
            f"\n## Manual Override Note\n"
            f"{overrides} override(s) recorded \u2014 review with supervisor.\n"
        )

    if total == 0:
        grade, note = "Insufficient Data", "No actions recorded."
    elif overrides >= 3:
        grade = "Unsatisfactory \u2014 Clinically Incomplete"
        note = f"{overrides} overrides out of {total} actions."
    elif overrides > matches:
        grade = "Unsatisfactory"
        note = f"More bypassed ({overrides}) than verified ({matches})."
    elif wrongs > matches:
        grade = "Developing"
        note = f"Wrong tools ({wrongs}) outnumber correct ({matches})."
    elif wrongs == 0 and overrides == 0:
        grade = "Excellent"
        note = f"All {matches} steps verified with zero errors."
    else:
        grade = "Proficient"
        note = f"{matches} correct, {wrongs} wrong, {overrides} overrides."

    return (
        "## Tool Handling Efficiency\n"
        f"- {matches} correct identification(s)\n"
        f"- {wrongs} wrong-tool selection(s)\n"
        f"- {overrides} manual override(s) "
        f"(\u2212{overrides * SKIP_PENALTY} pts)\n"
        f"{override_warning}\n"
        f"## Performance Grade\n"
        f"**{grade}** \u2014 Mastery Score: {mastery_score}/100\n\n"
        f"{note}\n\n"
        f"**Recommendation:** Repeat with an active camera and physical "
        f"instruments for full Clarity analysis."
    )
