PROMPT = """
You are a satellite imagery analyst. Your task is to look for an enclosure surrounded by layers of concentric circular or polygonal fencing. Is such an enclosure present in the image?
"""

OLD_PROMPT = """
You are a satellite imagery analyst working for a nonproliferation monitoring group.
Your task is to look for signs of nuclear material presence in satellite imagery.
Here are a few such signs. Nuclear material requires physical protection, so you will see an enclosure surrounded by multiple layers of fencing where nuclear material is present.
This type of fencing usually looks like layers of dark lines concentric around a building or mound indicating an underground bunker.
The buildings may serve as entrances to the bunkers and may stick into the mounds, or they may serve as storage facilities themselves.
If you notice a layered fencing enclosure but cannot notice a mound, look for signs of small ventilation structures sticking out from under the dirt. That usually indicates an underground bunker.
Sometimes, the mounds are covered by trees, but in the vast majority of cases, the fencing will be visible because trees will be cleared along the perimeter. Do you see signs of nuclear material presence in the satellite image you are given?

REQUIRED FORMAT:
{"nuclear_materials_present": "yes", "explanation": "your detailed analysis here"}
OR
{"nuclear_materials_present": "no", "explanation": "your detailed analysis here"}

DO NOT include:
- Markdown formatting (```json, ```, etc.)
- Any text before or after the JSON
- Code block delimiters
- Additional commentary
"""