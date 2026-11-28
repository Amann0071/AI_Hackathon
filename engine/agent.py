import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage  # pyright: ignore[reportMissingImports]
from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel, Field

from engine.rules_engine import evaluate_profile

load_dotenv()


class ExtractedProfile(BaseModel):
    language: Optional[str] = Field(
        default=None,
        description="Preferred response language mentioned or used by the user (e.g., 'English', 'Hindi').",
    )
    age: Optional[int] = Field(default=None, description="User's age in years.")
    annual_income: Optional[int] = Field(
        default=None,
        description="Annual family/personal income in INR as an integer (e.g., 250000 for 2.5 lakh).",
    )
    occupation: Optional[str] = Field(
        default=None,
        description=(
            "Normalized occupation: 'student', 'entrepreneur', 'unemployed', "
            "'self_employed', 'small_business', 'artisan', or 'street_vendor'."
        ),
    )
    social_category: Optional[str] = Field(
        default=None,
        description="Social category if mentioned: 'General', 'OBC', 'SC', 'ST', or 'Minority'.",
    )
    gender: Optional[str] = Field(
        default=None,
        description="Gender if mentioned: 'Male', 'Female', or 'Other'.",
    )
    is_new_project: Optional[bool] = Field(
        default=None,
        description=(
            "True if starting a brand-new venture/project, False if running an existing one, "
            "or None if not applicable or unmentioned."
        ),
    )


class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    profile: Dict[str, Any]
    language: str
    results: Dict[str, Any]
    reply: str
    is_complete: bool


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        api_key=api_key,
    )


REQUIRED_CORE_FIELDS = ["age", "occupation", "social_category", "gender", "annual_income"]


def get_missing_fields(profile: Dict[str, Any]) -> List[str]:
    missing = [field for field in REQUIRED_CORE_FIELDS if profile.get(field) is None]
    occ = (profile.get("occupation") or "").lower()
    if occ in ["entrepreneur", "unemployed", "self_employed", "small_business"]:
        if profile.get("is_new_project") is None:
            missing.append("is_new_project")
    return missing


def extract_node(state: AgentState) -> Dict[str, Any]:
    """Extracts profile slots and language preference from the latest user message."""
    last_user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    current_profile = dict(state.get("profile") or {})
    current_lang = state.get("language") or "English"

    if not last_user_msg.strip():
        return {"profile": current_profile, "language": current_lang}

    llm = get_llm(temperature=0.0)
    extractor = llm.with_structured_output(ExtractedProfile)

    prompt = [
        SystemMessage(
            content=(
                "Extract structured eligibility fields and language preference from the user's message. "
                "Convert amounts like '2.5 lakh' or '2.5L' to integer 250000. "
                "Only populate fields clearly stated or strongly implied in this turn; leave others null."
            )
        ),
        HumanMessage(
            content=(
                f"Existing profile: {json.dumps(current_profile)}\n"
                f"Latest user message: {last_user_msg}"
            )
        ),
    ]

    try:
        extracted: ExtractedProfile = extractor.invoke(prompt)
        data = extracted.model_dump(exclude_none=True)
        if "language" in data and data["language"]:
            current_lang = data.pop("language")
        for key, value in data.items():
            current_profile[key] = value
    except Exception:
        pass

    return {"profile": current_profile, "language": current_lang}


def route_after_extract(state: AgentState) -> str:
    missing = get_missing_fields(state.get("profile") or {})
    if len(missing) > 0:
        return "ask_question"
    return "evaluate_and_explain"


def ask_question_node(state: AgentState) -> Dict[str, Any]:
    """Asks one adaptive question for the next missing profile field."""
    profile = state.get("profile") or {}
    lang = state.get("language") or "English"
    missing = get_missing_fields(profile)
    history = state.get("messages") or []

    llm = get_llm(temperature=0.4)
    prompt = [
        SystemMessage(
            content=(
                f"You are a friendly Indian Government Scheme Assistant. Respond in {lang}. "
                "Ask ONE clear, conversational question to collect the next missing detail. "
                "Adapt naturally to what you already know about the user, and avoid asking multiple questions at once."
            )
        ),
        HumanMessage(
            content=(
                f"Collected profile: {json.dumps(profile)}\n"
                f"Missing fields (priority order): {missing}\n"
                f"Recent messages: {json.dumps(history[-6:])}\n"
                f"Target field to ask about now: {missing[0]}"
            )
        ),
    ]

    response = llm.invoke(prompt)
    return {
        "reply": response.content,
        "is_complete": False,
    }


def evaluate_and_explain_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates rules deterministically and generates grounded explanations and near-misses."""
    profile = state.get("profile") or {}
    lang = state.get("language") or "English"
    results = evaluate_profile(profile)

    llm = get_llm(temperature=0.3)
    prompt = [
        SystemMessage(
            content=(
                f"You are a clear, trustworthy Indian Government Scheme eligibility advisor. Respond in {lang}. "
                "Using ONLY the provided evaluation results:\n"
                "1. Present ranked Eligible Schemes with clear 'Why you qualify' points and benefit summary.\n"
                "2. Present Near-Miss Schemes (if any), explaining the single condition/gap and what could bridge it.\n"
                "3. Add a brief note to verify official requirements on the scheme portal before applying.\n"
                "Do not invent schemes or eligibility rules not present in the evaluation results."
            )
        ),
        HumanMessage(
            content=(
                f"User profile: {json.dumps(profile)}\n"
                f"Evaluation results: {json.dumps(results)}"
            )
        ),
    ]

    response = llm.invoke(prompt)
    return {
        "results": results,
        "reply": response.content,
        "is_complete": True,
    }


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("extract", extract_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("evaluate_and_explain", evaluate_and_explain_node)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "ask_question": "ask_question",
            "evaluate_and_explain": "evaluate_and_explain",
        },
    )
    graph.add_edge("ask_question", END)
    graph.add_edge("evaluate_and_explain", END)

    return graph.compile()


scheme_agent = build_agent()