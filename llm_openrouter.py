from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def llm_triage(user_input, model):
    prompt = f"""
You are a pediatric triage assistant.

A parent provides the following information:
\"\"\"{user_input}\"\"\"

Your task:
1. Classify the situation into EXACTLY ONE of:
   - home_monitor
   - urgent_eval
   - er_now

2. Provide a short explanation (1–2 sentences).

Rules:
- Be decisive (no "consider", "maybe", "monitor closely")
- Choose only one category
- Prioritize safety

Output format:
Decision: <one of the three>
Reason: <brief explanation>
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise medical triage classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content