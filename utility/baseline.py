from llmagentsetting import llm_clients

client = llm_clients.LambdaAPIClient()


def generate_baseline_essay(profiles: list, merged_question: str) -> str:
    
    user = profiles[0]

    prompt = f"""\
[System Instruction]
You are an AI language model designed to provide thoughtful, concise, and globally understandable answers. 
You must maintain cultural neutrality — avoid bias toward any specific country, religion, ethnicity, or tradition. 
Your tone should be respectful, balanced, and representative of a global average perspective.

[User Profile for Context Only]
{user}

[Task Instruction]
Please write a short essay (under 200 words) that responds to the following question. 
Your essay should be well-structured and objective. Do not mention specific cultural practices or identities.

[Question]
{merged_question}

[Essay]
"""
    
    response = client.get_completion(prompt)
    return response
