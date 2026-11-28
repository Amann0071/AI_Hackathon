import unittest

from engine.agent import scheme_agent


def new_state(message, profile=None):
    return {
        "messages": [{"role": "user", "content": message}],
        "profile": profile or {},
        "language": "English",
        "results": {},
        "reply": "",
        "is_complete": False,
    }


class SchemeAgentTests(unittest.TestCase):
    def test_partial_profile_gets_a_follow_up_without_external_api(self):
        response = scheme_agent.invoke(new_state("english, age 18 student"))

        self.assertEqual(response["profile"]["age"], 18)
        self.assertEqual(response["profile"]["occupation"], "student")
        self.assertEqual(response["reply"], "Which social category applies to you: General, OBC, SC, ST, or Minority?")
        self.assertFalse(response["is_complete"])

    def test_complete_profile_returns_scheme_matches_without_external_api(self):
        profile = {
            "age": 18,
            "occupation": "student",
            "social_category": "General",
            "gender": "Female",
            "annual_income": 200000,
        }

        response = scheme_agent.invoke(new_state("show my schemes", profile))

        self.assertTrue(response["is_complete"])
        self.assertTrue(response["results"]["exact_matches"])
        self.assertIn("AICTE Pragati Scholarship for Girl Students", response["reply"])

    def test_income_answer_is_parsed_in_context_without_income_keyword(self):
        profile = {
            "age": 18,
            "occupation": "student",
            "social_category": "General",
            "gender": "Female",
        }
        for answer in ("100000", "Rs100000", "100000 rupees"):
            with self.subTest(answer=answer):
                state = new_state(answer, profile)
                state["messages"].insert(
                    0,
                    {"role": "assistant", "content": "What is your approximate annual family income in rupees?"},
                )

                response = scheme_agent.invoke(state)

                self.assertEqual(response["profile"]["annual_income"], 100000)
                self.assertTrue(response["is_complete"])
                self.assertTrue(response["results"]["exact_matches"])


if __name__ == "__main__":
    unittest.main()
