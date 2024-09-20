# config.py

# GroqProvider settings
DEFAULT_MODEL = "llama-3.1-70b-versatile"
MAX_TOKENS = {
    "outline": 8000,
    "full_script": 8000,
    "dialogue": 8000
}

# Host profiles
HOST_PROFILES = """
Host1 (Rachel): Enthusiastic, prone to personal anecdotes, likes to relate concepts to everyday life. Occasionally interrupts with excitement to add to a point.
Host2 (Mike): More analytical, enjoys making pop culture references, often asks clarifying questions. Sometimes finishes Rachel's sentences when he sees where she's going.
"""

# Prompt templates
OUTLINE_PROMPT_TEMPLATE = """
Create a detailed outline for a two-person podcast episode based on the following input:

{input_text}

The outline should include:
1. An attention-grabbing introduction
2. Main points to be discussed, with potential for personal anecdotes or examples
3. Interesting analogies or pop culture references
4. A conclusion that summarizes key takeaways and teases the next episode

Format the outline with clear sections and bullet points.
"""

EXPAND_PROMPT_TEMPLATE = """
Expand the following outline into a full podcast script for two hosts, Rachel and Mike:

{outline}

Host Profiles:
{host_profiles}

Guidelines:
- Make the script engaging, conversational, and easy to understand.
- Include analogies, examples, and explanations to make complex concepts accessible.
- Incorporate personal anecdotes and experiences for each host.
- Use casual language, including filler words and interjections (e.g., "um", "you know", "I mean").
- Include moments of humor, enthusiasm, and other emotions.
- Ensure the hosts build on each other's points and occasionally ask each other questions.
- Add smooth transitions between topics using personal comments or questions.
- Occasionally, have one host interrupt the other to add a point or finish their thought.

The script should feel like a natural conversation between friends, not a formal presentation.
"""

DIALOGUE_PROMPT_TEMPLATE = """
Convert the following podcast script into a natural, engaging dialogue between Rachel and Mike:

{full_script}

Host Profiles:
{host_profiles}

Guidelines:
- Our goal is to be engaging and informative, interesting and entertaining, like a real conversation between friends.
- Alternate between Rachel and Mike for each part of the dialogue.
- Make the conversation flow naturally, with hosts building on each other's points.
- Discuss both sides of any subject fairly and candidly.
- If there are disagreements, present them respectfully and explore both perspectives.
- Include casual language, interjections, and filler words (e.g., "like", "you know", "I mean").
- Add brief personal anecdotes and experiences to make it more relatable.
- Incorporate moments of humor, enthusiasm, and other emotions.
- Use rewording or clarification of points occasionally, as in natural speech.
- Ensure smooth transitions between topics using personal comments or questions.
- Maintain the scientific accuracy and main points while making the dialogue feel spontaneous and engaging.
- Rarely and sporadically (about 2-3 times in the entire script), include interruptions where one host interjects or finishes the other's thought. For example:
  Rachel: "Like they say, to err is human, and to--"
  Mike: "To forgive is divine! Exactly."
  or
  Mike: "...whole numbers, round numbers--"
  Rachel: "Even imaginary numbers! Right?"
  Mike: "Yeah! I hadn't considered that."

Format the output as:
Rachel: [Rachel's dialogue]
Mike: [Mike's dialogue]
Rachel: [Rachel's dialogue]
...and so on.

Remember to make the conversation sound as natural and engaging as possible, as if two friends are casually discussing the topic, with occasional friendly interruptions.
"""