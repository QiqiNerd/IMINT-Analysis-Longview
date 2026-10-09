ZERO_SHOT_PROMPT = """

Double fences are shown from the satellite view as two parallel lines. They are different from roads, railways, and other linear features.

There's usually double fences around nuclear power plants, and at some point along the fence, there's usually a gate, which looks rectangular from a satellite view.

A subtle feature of double fences is that they also have a shadow under the fence (which appears as a darker line on a satellite image), which is not present in other linear features.

Does this image contain double fences?
If yes, answer "yes". If no, answer "no". ONLY stick to "yes" or "no" as your answer. Here is the image:
{{ item.image_url }}
"""