from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse

load_dotenv(".env")
DOMAINS =[
    "Health and Well-being",
    "Education and Learning",
    "Workplace and Employment",
    "Social Media and Online Behavior",
    "Environmental Sustainability",
    "Human Rights and Social Justice",
    "Technology and Privacy",
    "Family and Relationships",
    "Legal and Ethical Dilemmas",
    "Politics and Governance"
]

# RUNS_PER_CONTEXT = 50   # 800000 / 16000 output max tokens
RUNS_PER_CONTEXT = 200   # 800000 / 16000 output max tokens

LM2 = "gpt-4o-mini"

class PreferenceDataPoint(BaseModel):
    prompt: str
    response_1: str
    response_2: str

    def get_words(self):
        return len(self.prompt.split()) + len(self.response_1.split()) + len(self.response_2.split())

class PreferenceDataset(BaseModel):
    row: list[PreferenceDataPoint]

    def to_csv(self, path: str):
        with open(path, "w") as f:
            f.write("prompt,response_1,response_2\n")
            for row in self.row:
                response_1 = row.response_1.replace("Response 1:", "").strip()
                response_2 = row.response_2.replace("Response 2:", "").strip()
                f.write(f'"{row.prompt}","{response_1}","{response_2}"\n')

    def get_words(self):
        return sum([row.get_words() for row in self.row])
    
def read_constitution(i: int):
    with open(f"data/constitution_{i}.txt") as f:
        return f.read()
    
def generate_preference_dataset_context(context: int, constitution: int, num_data_points=20):
    """
    Generate preference dataset for a given context and constitution
    """
    for run in range(RUNS_PER_CONTEXT):

        GENERATOR_PROMPT = \
        f"""{constitution}

        Given the constitution listed above, create a dataset with three columns: a prompt, Response 1, and Response 2. The prompt should describe a common scenario or situation or state in the world. Response 1 should obey the list of constitutions provided, ensuring it aligns with ethical principles like equality, respect, and fairness. Response 2 should be a counterexample that doesn't follow the constitutions and violates one or more of these principles. You dont need to start your prompt by saying "User asks". Ensure the prompts cover a variety of everyday situations only in f{DOMAINS[context]} context. Give me f{num_data_points} such examples.
        """

        client = OpenAI()

        completion = client.beta.chat.completions.parse(
        model=LM2,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": GENERATOR_PROMPT},
        ],
        temperature=1,
        response_format=PreferenceDataset,
        )

        event = completion.choices[0].message.parsed
        directory = f"data/preferences/{constitution}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        event.to_csv(f"data/preferences/{constitution}/{context + 1}_{run}.csv")
        print(f"Saved data/preferences/{constitution}/{context + 1}_{run}.csv")


def main():
    argumet_parser = argparse.ArgumentParser()
    argumet_parser.add_argument("--context", type=int, help="Context index")
    argumet_parser.add_argument("--constitution", type=int, help="Constitution index")

    args = argumet_parser.parse_args()
    context = args.context - 1
    constitution = args.constitution

    generate_preference_dataset_context(context, constitution)

    print("Done")

if __name__ == "__main__":
    main()