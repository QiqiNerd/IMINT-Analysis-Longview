PROMPT = """
Would you guess this image contains an anti-aircraft artillery (AAA) site?
Respond with JSON in this format:
{
  "aaa_present": "yes" or "no",
  "explanation": "detailed reasoning for your decision"
}

Now classify this image:
"""

PROMPT_WITH_GUIDANCE = """
AAA sites are usually in a star-like formation, or a straight light formation. Burial mounds in North Korea are often mistaken for AAA sites,
and you can distinguish them because they are disconnected and not orderly arranged.

Would you guess this image contains an anti-aircraft artillery (AAA) site?
If yes, answer "yes" and explain what you observe (e.g., "star-like formation of circular structures", "linear arrangement of defensive positions", etc.).
If no, answer "no" and explain why you don't see AAA sites (e.g., "only burial mounds visible, disconnected and not orderly", "residential area with no defensive structures", etc.).

Respond with JSON in this format:
{
  "aaa_present": "yes" or "no",
  "explanation": "detailed reasoning for your decision"
}

Now classify this image:
"""