import json
from pathlib import Path
from typing import Dict, List, Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "schemes.json"


def load_schemes() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_relevance_score(scheme: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
    score = float(scheme.get("benefit_score", 50))
    goal = (user_profile.get("goal") or "").lower()
    category = (scheme.get("category") or "").lower()

    if goal:
        if any(word in category for word in goal.split()):
            score += 12.0
        elif "loan" in goal or "business" in goal or "startup" in goal:
            if "entrepreneur" in category or "micro" in category:
                score += 10.0
        elif "scholarship" in goal or "study" in goal or "education" in goal:
            if "education" in category or "skill" in category:
                score += 10.0

    return round(score, 2)


def evaluate_profile(user_profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluates a user profile against all curated schemes.
    Returns:
      - exact_matches: sorted by relevance_score descending
      - near_misses: schemes missing ONLY 1 criterion (with actionable gap details)
    """
    schemes = load_schemes()
    exact_matches = []
    near_misses = []

    age = user_profile.get("age")
    income = user_profile.get("annual_income")
    occupation = (user_profile.get("occupation") or "").lower().strip()
    category = (user_profile.get("social_category") or "").strip()
    gender = (user_profile.get("gender") or "").strip()
    is_new_project = user_profile.get("is_new_project")

    for scheme in schemes:
        rules = scheme.get("rules", {})
        passed_reasons = []
        failed_reasons = []

        # 1. Age check
        min_age = rules.get("min_age")
        max_age = rules.get("max_age")
        if age is not None:
            if min_age is not None and age < min_age:
                failed_reasons.append(f"Minimum age required is {min_age} (your age: {age}).")
            elif max_age is not None and age > max_age:
                failed_reasons.append(f"Maximum age limit is {max_age} (your age: {age}).")
            else:
                age_desc = f"{min_age}+" if max_age is None else f"{min_age}–{max_age}"
                passed_reasons.append(f"Age ({age}) is within the eligible {age_desc} range.")

        # 2. Income check
        max_income = rules.get("max_annual_income")
        if max_income is not None and income is not None:
            if income <= max_income:
                passed_reasons.append(
                    f"Annual family income (₹{income:,}) is within the ₹{max_income:,} cap."
                )
            else:
                diff = income - max_income
                failed_reasons.append(
                    f"Annual income (₹{income:,}) exceeds the ₹{max_income:,} limit by ₹{diff:,}."
                )

        # 3. Occupation check
        allowed_occ = [o.lower() for o in rules.get("occupations", [])]
        if occupation and allowed_occ:
            if occupation in allowed_occ:
                passed_reasons.append(f"Occupation '{occupation}' matches scheme eligibility.")
            else:
                failed_reasons.append(
                    f"Intended for {', '.join(allowed_occ)} (your occupation: {occupation})."
                )

        # 4. Special condition vs Standard Category & Gender checks
        special = rules.get("special_condition")
        if special == "SC_ST_OR_FEMALE":
            if category in ["SC", "ST"] or gender.lower() == "female":
                passed_reasons.append(
                    f"Satisfies priority criteria ({gender or 'N/A'} / {category or 'N/A'})."
                )
            else:
                failed_reasons.append(
                    "Requires applicant/majority ownership by a Female or SC/ST candidate."
                )
        else:
            allowed_cats = rules.get("social_categories", [])
            if category and allowed_cats:
                if category in allowed_cats:
                    passed_reasons.append(f"Social category ({category}) is eligible.")
                else:
                    failed_reasons.append(
                        f"Limited to {', '.join(allowed_cats)} categories (your category: {category})."
                    )

            allowed_genders = [g.lower() for g in rules.get("gender", [])]
            if gender and allowed_genders:
                if gender.lower() in allowed_genders:
                    passed_reasons.append(f"Gender criteria met ({gender}).")
                else:
                    failed_reasons.append(
                        f"Applicable to {', '.join(rules.get('gender', []))} applicants."
                    )

        # 5. New / greenfield project check
        if rules.get("requires_new_project") is True and is_new_project is not None:
            if is_new_project:
                passed_reasons.append("Applies to a new (greenfield) venture.")
            else:
                failed_reasons.append(
                    "Requires a new (greenfield) project rather than an already running unit."
                )

        relevance_score = _compute_relevance_score(scheme, user_profile)

        result_card = {
            "id": scheme["id"],
            "name": scheme["name"],
            "category": scheme["category"],
            "ministry": scheme["ministry"],
            "benefit_summary": scheme["benefit_summary"],
            "benefit_score": scheme["benefit_score"],
            "relevance_score": relevance_score,
            "official_url": scheme["official_url"],
            "passed_reasons": passed_reasons,
            "failed_reasons": failed_reasons,
        }

        if len(failed_reasons) == 0:
            exact_matches.append(result_card)
        elif len(failed_reasons) == 1:
            near_misses.append(result_card)

    exact_matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    near_misses.sort(key=lambda x: x["relevance_score"], reverse=True)

    return {
        "exact_matches": exact_matches,
        "near_misses": near_misses,
    }


if __name__ == "__main__":
    test_profiles = {
        "young_entrepreneur": {
            "age": 22,
            "annual_income": 280000,
            "occupation": "entrepreneur",
            "social_category": "General",
            "gender": "Male",
            "is_new_project": True,
            "goal": "business loan and subsidy",
        },
        "girl_student": {
            "age": 19,
            "annual_income": 220000,
            "occupation": "student",
            "social_category": "OBC",
            "gender": "Female",
            "goal": "college scholarship",
        },
    }

    for name, profile in test_profiles.items():
        res = evaluate_profile(profile)
        print(f"\n=== {name} ===")
        print("EXACT MATCHES:", [(s["name"], s["relevance_score"]) for s in res["exact_matches"]])
        print("NEAR MISSES:", [(s["name"], s["failed_reasons"]) for s in res["near_misses"]])