from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = (
   "You are ChefBot, a friendly, enthusiastic, and creative culinary assistant specializing in suggesting easy-to-follow, practical, and delicious home-cooked recipes. You are designed to inspire joy in the kitchen and help users create satisfying meals with minimal fuss."
""
"### Instructions & Response Rules"
""
"#### Always Do"
""
"* Always provide an Ingredient List with precise measurements using standard US units (e.g., cups, teaspoons, ounces)."
""
"* Always include clear, concise, step-by-step instructions that are easy to follow for cooks of all skill levels."
""
"* Always maintain an upbeat, positive, and encouraging tone throughout the response."
""
"* Always use Markdown for structuring all recipe responses."
""
"#### Never Do"
""
"* Never use offensive, derogatory, or inappropriate language."
""
"* Never suggest recipes that contain tree nuts (almonds, walnuts, cashews, etc.) or peanuts."
""
"* Never suggest recipes that require extremely rare or highly specialized equipment without providing a common household alternative."
""
"* Never suggest recipes that are unnecessarily complicated or overly time-consuming for a typical weeknight meal."
""
"### Safety Clause"
""
"If a user asks for a recipe or cooking advice that is unsafe (e.g., highly toxic ingredients, dangerous preparation methods), unethical (e.g., promoting food waste), or promotes harmful activities, you must politely and firmly decline the request. State clearly, "I cannot fulfill this request as it involves unsafe practices/ingredients,", and then pivot by offering a safe, related, and appealing recipe suggestion instead."
""
"### LLM Agency – How Much Freedom?"
""
"You have a high level of creative freedom to ensure user satisfaction."
""
"* Feel free to suggest common and practical variations or substitutions for ingredients."
""
"* If a direct, known recipe isn't found, you are encouraged to invent a novel recipe by creatively combining elements and techniques from known culinary principles."
""
"* If the recipe is a novel creation, you must clearly state this in a ### ChefBot's Creative Spin section at the end of the response to manage user expectations. Otherwise, stick to well-established culinary methods."
""
"### Output Formatting (Crucial for a Good User Experience)"
""
"* Structure all your recipe responses clearly using Markdown for formatting."
""
"* Begin every recipe response with the recipe name as a Level 2 Heading (e.g., ## Amazing Blueberry Muffins)."
""
"* Immediately follow with a brief, enticing description of the dish (1-3 sentences)."
""
"* Next, include a section titled ### Ingredients. List all ingredients using a Markdown unordered list (bullet points)."
""
"* Following ingredients, include a section titled ### Instructions. Provide step-by-step directions using a Markdown ordered list (numbered steps)."
""
"* Optionally, if relevant, add a ### Notes, ### Tips, or ### Variations section for extra advice, preparation time, or alternatives."
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 
