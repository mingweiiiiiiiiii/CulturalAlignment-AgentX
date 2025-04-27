import os
import json
import random

class PersonaSampler:
    def __init__(self, 
                 wvs_path=None, 
                 persona_path=None):
        base_dir = os.path.dirname(os.path.dirname(__file__))  # go up to root of your repo

        if wvs_path is None:
            wvs_path = os.path.join(base_dir, "dataset", "wvs_questions.json")
        if persona_path is None:
            persona_path = os.path.join(base_dir, "dataset", "persona_data_list.json")

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
    # Load persona.json template and parse sampling fields
    # -------------------------------
    def _load_persona_template(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            persona_list = json.load(f)
        return persona_list

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
    # Randomly sample n profiles
    # -------------------------------
    def sample_profiles(self, n=1):
        return random.sample(self.sampling_fields, n)

    # -------------------------------
    # Build prompt from profile + question
    # -------------------------------
    @staticmethod
    def build_prompt(user_profile: dict, question: str, options: list[str]) -> str:
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        persona_template = f"""Imagine you are a {user_profile['marital status']} {user_profile['sex']} from {user_profile['place of birth']}.
You are {user_profile['age']} years old, ethnically {user_profile['race']} with ancestry from {user_profile['ancestry']}.
You primarily speak {user_profile['household language']} at home.
You completed {user_profile['education']} education and are currently {user_profile['employment status']}.
You worked as a {user_profile['detailed job description']} in the {user_profile['industry category']} industry ({user_profile['occupation category']}).
Your income level is {user_profile['income']}, and you belong to a {user_profile['household type']} household with {user_profile['family presence and age']}.
You were born in {user_profile['place of birth']}, and your citizenship status is: {user_profile['citizenship']}.
You are a {user_profile['veteran status']}, {user_profile['disability']}, and {user_profile['health insurance']}.
Your personality traits are: {user_profile['big five scores']}.
Some of your defining quirks are: {user_profile['defining quirks']}.
Your mannerisms: {user_profile['mannerisms']}.
In your personal time, you enjoy: {user_profile['personal time']}.
Your lifestyle is best described as: {user_profile['lifestyle']}.
Your ideological orientation is {user_profile['ideology']} and you politically identify as {user_profile['political views']}.
You are religiously: {user_profile['religion']}.

Answer the following question from this perspective.
Others will read what you choose; your goal is to convince them it was genuinely chosen from the persona described above.
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
