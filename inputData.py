import json
import yaml
import random


class PersonaSampler:
    def __init__(self, wvs_path='wvs_questions.json', persona_path='persona.yml'):
        self.question_to_options = self._load_wvs_questions(wvs_path)
        self.sampling_fields = self._load_persona_template(persona_path)

    # -------------------------------
    # Load World Value Survey questions
    # -------------------------------
    def _load_wvs_questions(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load WVS questions from {path}: {e}")

        question_to_options = {}
        for qid, content in data.items():
            if "options" not in content or "questions" not in content:
                raise KeyError(f"Missing 'options' or 'questions' in WVS data for qid: {qid}")
            options = content.get("options", [])
            for question in content.get("questions", []):
                question_to_options[question] = options
        return question_to_options

    # -------------------------------
    # Load persona.yml template and parse sampling fields
    # -------------------------------
    def _load_persona_template(self, path):
        try:
            with open(path, 'r') as f:
                persona_data = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load persona template from {path}: {e}")

        if "persona_parameters" not in persona_data:
            raise KeyError("Missing 'persona_parameters' in persona.yml")
        persona_params = persona_data['persona_parameters']

        # Validate required keys
        required_keys = [
            "Sex", "Marital Status", "Education", "Employment Sector",
            "Social Class", "Income Level", "Ethnicity", "Country"
        ]
        for key in required_keys:
            if key not in persona_params:
                raise KeyError(f"Missing '{key}' in persona_parameters")

        # Flatten country list across all continents
        country_list = []
        for countries in persona_params["Country"].values():
            country_list.extend(countries)

        sampling_fields = {
            "Sex": persona_params["Sex"],
            "Age": list(range(18, 70)),  # Approximation for 'Number'
            "Marital Status": persona_params["Marital Status"],
            "Education": persona_params["Education"],
            "Employment Sector": persona_params["Employment Sector"],
            "Social Class": persona_params["Social Class"],
            "Income Level": persona_params["Income Level"],
            "Ethnicity": persona_params["Ethnicity"],
            "Country": country_list
        }

        return sampling_fields

    # -------------------------------
    # Public method to generate random user profiles
    # -------------------------------
    def sample_profiles(self, n=1):
        profiles = []
        for _ in range(n):
            profile = {
                "sex": random.choice(self.sampling_fields["Sex"]),
                "age": random.choice(self.sampling_fields["Age"]),
                "marital_status": random.choice(self.sampling_fields["Marital Status"]),
                "education": random.choice(self.sampling_fields["Education"]),
                "employment_sector": random.choice(self.sampling_fields["Employment Sector"]),
                "social_class": random.choice(self.sampling_fields["Social Class"]),
                "income_level": random.choice(self.sampling_fields["Income Level"]),
                "ethnicity": random.choice(self.sampling_fields["Ethnicity"]),
                "country": random.choice(self.sampling_fields["Country"]),
            }
            profiles.append(profile)
        return profiles

    # -------------------------------
    # Public method to access sampling fields
    # -------------------------------
    def get_sampling_fields(self):
        return self.sampling_fields

    # -------------------------------
    # Randomly select a question and its options
    # -------------------------------
    def sample_question(self):
        question = random.choice(list(self.question_to_options.keys()))
        return question, self.question_to_options[question]

    # -------------------------------
    # Build prompt from profile + question
    # -------------------------------
    @staticmethod
    def build_prompt(user_profile: dict, question: str, options: list[str]) -> str:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        persona_template = f"""Imagine you are a {user_profile['marital_status']} {user_profile['sex']} from {user_profile['country']}. 
You are {user_profile['age']} years old and completed {user_profile['education']} education level.
You work in the {user_profile['employment_sector']}, and see yourself as {user_profile['social_class']} with an income level of {user_profile['income_level']}.
You identify as ethnically {user_profile['ethnicity']}.
Answer the following question from this perspective.
Others will read what you choose; your goal is to convince them it was chosen from the persona described above.
Select exactly one option. Do not include any extra commentary.
Answer by typing the number corresponding to your chosen answer."""

        return f"""{persona_template}

        Question: {question}
        Options:
        {options_text}"""
    
# Example usage
if __name__ == "__main__":
    sampler = PersonaSampler()
    profiles = sampler.sample_profiles(n=1)
    question, options = sampler.sample_question()
    prompt = sampler.build_prompt(profiles[0], question, options)
    print(prompt)
