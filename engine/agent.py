import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingImports]

from engine.rules_engine import evaluate_profile


class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    profile: Dict[str, Any]
    language: str
    results: Dict[str, Any]
    reply: str
    is_complete: bool


REQUIRED_CORE_FIELDS = ["age", "occupation", "social_category", "gender", "annual_income"]


def get_missing_fields(profile: Dict[str, Any]) -> List[str]:
    missing = [field for field in REQUIRED_CORE_FIELDS if profile.get(field) is None]
    occupation = (profile.get("occupation") or "").lower()
    if occupation in ["entrepreneur", "unemployed", "self_employed", "small_business"]:
        if profile.get("is_new_project") is None:
            missing.append("is_new_project")
    return missing


def _extract_income(text: str, allow_bare_amount: bool = False) -> int | None:
    match = re.search(
        r"\b(?:annual|family|household|yearly)?\s*income\b.{0,24}?"
        r"(?:₹|rs\.?\s*)?([\d,]+(?:\.\d+)?)\s*"
        r"(lakh|lakhs|lac|lacs|l|crore|crores|cr|thousand|k|rupees?|rs\.?)?\b",
        text,
        re.IGNORECASE,
    )
    if not match and allow_bare_amount:
        match = re.fullmatch(
            r"\s*(?:₹|rs\.?\s*)?([\d,]+(?:\.\d+)?)\s*"
            r"(lakh|lakhs|lac|lacs|l|crore|crores|cr|thousand|k|rupees?|rs\.?)?\s*",
            text,
            re.IGNORECASE,
        )
    if not match:
        return None

    amount = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "").lower()
    if unit in {"lakh", "lakhs", "lac", "lacs", "l"}:
        amount *= 100_000
    elif unit in {"crore", "crores", "cr"}:
        amount *= 10_000_000
    elif unit in {"thousand", "k"}:
        amount *= 1_000
    return int(amount)


def _extract_node(state: AgentState) -> Dict[str, Any]:
    last_user_message = next(
        (
            message.get("content", "")
            for message in reversed(state.get("messages", []))
            if message.get("role") == "user"
        ),
        "",
    )
    text = last_user_message.strip()
    lowered = text.lower()
    profile = dict(state.get("profile") or {})
    language = state.get("language") or "English"
    missing_before = get_missing_fields(profile)
    expected_field = missing_before[0] if missing_before else None

    age_match = re.search(
        r"\b(?:age|aged)\s*(?:is|:)?\s*(\d{1,3})\b|\bi am\s+(\d{1,3})\b",
        lowered,
    )
    if not age_match and expected_field == "age":
        age_match = re.fullmatch(r"\s*(\d{1,3})(?:\s*(?:years?|yrs?|yo))?\s*", lowered)
    if age_match:
        age = int(age_match.group(1) or age_match.group(2))
        if 0 <= age <= 120:
            profile["age"] = age

    occupations = [
        (r"\b(?:small business|small-business)\b", "small_business"),
        (r"\b(?:self[- ]employed|self employed)\b", "self_employed"),
        (r"\b(?:street vendor|street-vendor|hawker)\b", "street_vendor"),
        (r"\b(?:entrepreneur|business owner|starting a business)\b", "entrepreneur"),
        (r"\b(?:student|pupil|college student)\b", "student"),
        (r"\b(?:unemployed|job seeker|looking for work)\b", "unemployed"),
        (r"\b(?:artisan|craftsperson|craft worker)\b", "artisan"),
        (r"\bfarmer\b", "farmer"),
    ]
    for pattern, occupation in occupations:
        if re.search(pattern, lowered):
            profile["occupation"] = occupation
            break

    category_match = re.search(r"\b(sc|st|obc|minority|general)\b", lowered)
    if category_match:
        profile["social_category"] = category_match.group(1).upper()
        if profile["social_category"] == "GENERAL":
            profile["social_category"] = "General"

    if re.search(r"\b(?:female|woman|women|girl)\b", lowered):
        profile["gender"] = "Female"
    elif re.search(r"\b(?:male|man|men|boy)\b", lowered):
        profile["gender"] = "Male"
    elif re.search(r"\b(?:nonbinary|non-binary|other gender)\b", lowered):
        profile["gender"] = "Other"

    income = _extract_income(lowered, allow_bare_amount=expected_field == "annual_income")
    if income is not None:
        profile["annual_income"] = income

    if re.search(r"\b(?:new project|new business|starting a new|brand.new)\b", lowered):
        profile["is_new_project"] = True
    elif re.search(r"\b(?:existing business|already running|not a new)\b", lowered):
        profile["is_new_project"] = False
    elif expected_field == "is_new_project":
        if re.fullmatch(r"\s*(?:yes|y|new)\s*[.!]?\s*", lowered):
            profile["is_new_project"] = True
        elif re.fullmatch(r"\s*(?:no|n|existing)\s*[.!]?\s*", lowered):
            profile["is_new_project"] = False

    if re.search(r"[\u0900-\u097f]|\bhindi\b|\bहिंदी\b", lowered):
        language = "Hindi"
    elif re.search(r"\benglish\b", lowered):
        language = "English"

    return {"profile": profile, "language": language}


QUESTIONS = {
    "age": ("What is your age in years?", "आपकी उम्र कितनी है?"),
    "occupation": (
        "What do you currently do (for example, student, farmer, entrepreneur, or looking for work)?",
        "आप अभी क्या काम करते हैं (जैसे विद्यार्थी, किसान, उद्यमी या नौकरी की तलाश में)?",
    ),
    "social_category": (
        "Which social category applies to you: General, OBC, SC, ST, or Minority?",
        "आपकी सामाजिक श्रेणी क्या है: General, OBC, SC, ST या Minority?",
    ),
    "gender": ("What is your gender?", "आपका लिंग क्या है?"),
    "annual_income": (
        "What is your approximate annual family income in rupees?",
        "आपकी परिवार की अनुमानित वार्षिक आय कितनी है?",
    ),
    "is_new_project": (
        "Is this a new project/business, or are you already running it?",
        "क्या यह नया व्यवसाय/परियोजना है या पहले से चल रहा है?",
    ),
}


def _ask_question_node(state: AgentState) -> Dict[str, Any]:
    missing = get_missing_fields(state.get("profile") or {})
    if not missing:
        return {"reply": "Please share one more detail so I can check your eligibility.", "is_complete": False}

    language_index = 1 if (state.get("language") or "").lower().startswith("hindi") else 0
    return {
        "reply": QUESTIONS[missing[0]][language_index],
        "is_complete": False,
    }


def _explain_gap(reason: str) -> str:
    occupation_match = re.search(r"Intended for (.+?) \(your occupation: (.+?)\)\.", reason)
    if occupation_match:
        eligible_occupations = occupation_match.group(1).replace("_", " ")
        current_occupation = occupation_match.group(2).replace("_", " ")
        return (
            f"Occupation criterion: applicants must be one of {eligible_occupations}. "
            f"Your profile lists {current_occupation}, which is not included. "
            "This near-match applies only if you also meet one of the listed occupation requirements."
        )

    category_match = re.search(r"Limited to (.+?) categories \(your category: (.+?)\)\.", reason)
    if category_match:
        return (
            f"Social-category requirement: applicants must be in {category_match.group(1)}. "
            f"Your profile says {category_match.group(2)}."
        )

    gender_match = re.search(r"Applicable to (.+?) applicants\.", reason)
    if gender_match:
        return f"Gender requirement: applicants must be {gender_match.group(1)}."

    minimum_age = re.search(r"Minimum age required is (\d+)", reason)
    if minimum_age:
        return f"Age requirement: applicants must be at least {minimum_age.group(1)} years old."

    maximum_age = re.search(r"Maximum age limit is (\d+)", reason)
    if maximum_age:
        return f"Age requirement: applicants must be no older than {maximum_age.group(1)} years."

    income_limit = re.search(r"exceeds the ₹([\d,]+) limit", reason)
    if income_limit:
        return f"Income requirement: annual income must not exceed ₹{income_limit.group(1)}."

    if "Requires a new (greenfield) project" in reason:
        return "Project requirement: the business must be a new (greenfield) project."

    return f"Eligibility requirement: {reason}"


def _evaluate_and_explain_node(state: AgentState) -> Dict[str, Any]:
    profile = state.get("profile") or {}
    results = evaluate_profile(profile)
    exact_matches = results["exact_matches"]
    near_misses = results["near_misses"]
    sections = []

    if exact_matches:
        lines = ["Eligible schemes:"]
        for scheme in exact_matches[:5]:
            lines.append(f"- {scheme['name']}: {scheme['benefit_summary']}")
            lines.append(f"  Official details: {scheme['official_url']}")
        sections.append("\n".join(lines))
    else:
        sections.append("No exact matches found for the details provided.")

    if near_misses:
        lines = ["Potential near-matches:"]
        for scheme in near_misses[:5]:
            gap = scheme["failed_reasons"][0] if scheme["failed_reasons"] else "One eligibility detail needs checking."
            scheme["eligibility_guidance"] = _explain_gap(gap)
            lines.append(f"- {scheme['name']}")
            lines.append(f"  What you need to meet: {scheme['eligibility_guidance']}")
            lines.append(f"  Official details: {scheme['official_url']}")
        sections.append("\n".join(lines))

    sections.append("Please verify current eligibility and application details on the official scheme portal.")
    return {
        "results": results,
        "reply": "\n\n".join(sections),
        "is_complete": True,
    }


def _route_after_extract(state: AgentState) -> str:
    if get_missing_fields(state.get("profile") or {}):
        return "ask_question"
    return "evaluate_and_explain"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("extract", _extract_node)
    graph.add_node("ask_question", _ask_question_node)
    graph.add_node("evaluate_and_explain", _evaluate_and_explain_node)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges(
        "extract",
        _route_after_extract,
        {
            "ask_question": "ask_question",
            "evaluate_and_explain": "evaluate_and_explain",
        },
    )
    graph.add_edge("ask_question", END)
    graph.add_edge("evaluate_and_explain", END)
    return graph.compile()


scheme_agent = build_agent()
