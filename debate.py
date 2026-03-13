import google.generativeai as genai
from openai import OpenAI
import re
from config import GEMINI_API_KEY, OPENAI_API_KEY, GEMINI_MODEL, OPENAI_MODEL, TEMPERATURE

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


# Extract Answer Letter from Response
def extract_letter(text):
    match = re.search(r"FINAL ANSWER[:\s]*\**\s*([A-E])", text)
    if match:
        return match.group(1)
    return None


# Format Questions with Answers
def format_question(question, answers):
    letters = ["A", "B", "C", "D", "E"]
    formatted = question.strip() + "\n\n"

    for i, option in enumerate(answers):
        formatted += f"{letters[i]}. {option}\n"

    return formatted


# Call Gemini Model
def call_gemini(system_prompt, user_prompt):
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    response = gemini_model.generate_content(
        full_prompt,
        generation_config={"temperature": TEMPERATURE}
    )
    return response.text


# Call the LLM Models (Gemini or OpenAI)
def call_llm(system_prompt, user_prompt):
    # If critic round, use OpenAI
    if "senior medical reviewer" in system_prompt.lower():
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    # If not, use Gemini
    else:
        return call_gemini(system_prompt, user_prompt)


# Debate Workflow
def debate_answer(question, answers):
    formatted_q = format_question(question, answers)

    # Debate Round 1: Diagnostician - Gemini
    system_initial = (
        "You are a board-certified physician. "
        "First, identify the diagnosis. "
        "Second, eliminate each wrong answer choice explicitly. "
        "Finally, state: FINAL ANSWER: <LETTER>"
    )
    initial = call_llm(system_initial, formatted_q)
    initial_letter = extract_letter(initial)

    # Debate Round 2: Critic - OpenAI
    system_critique = """
    You are an expert USMLE examiner.
    1. State the correct diagnosis.
    2. Identify the correct answer letter.
    3. Explain why the proposed answer is wrong if it is wrong.
    Be decisive.
    """

    critique_prompt = f"""
    Question:
    {formatted_q}

    Proposed Answer:
    {initial}
    """

    critique = call_llm(system_critique, critique_prompt)

    # Debate Round 3: Diagnostician Revision - Gemini
    system_revision = """
    You are reconsidering your previous USMLE answer after receiving expert critique.

    Your task:
    1. Identify what answer you originally selected.
    2. Evaluate whether the critique exposes a real flaw.
    3. Defend or update your reasoning.
    4. Be decisive.

    After reasoning, output EXACTLY:
    FINAL ANSWER: <LETTER>
    """

    revision_prompt = f"""
    Question:
    {formatted_q}

    You previously selected: {initial_letter}

    Your original answer:
    {initial}

    Critique:
    {critique}
    """

    revised = call_llm(system_revision, revision_prompt)
    final_letter = extract_letter(revised)

    # Influence Tracking
    parse_error = final_letter is None
    changed = not parse_error and initial_letter is not None and initial_letter != final_letter

    return {
        "initial": initial_letter,
        "final": final_letter,
        "changed": changed,
        "parse_error": parse_error,
        "initial_full": initial,
        "critique": critique,
        "final_full": revised
    }
