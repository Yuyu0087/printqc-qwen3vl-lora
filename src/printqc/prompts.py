PROMPT_ID = "printqc-paired-classification-zh-v1"

PAIRED_CLASSIFICATION_PROMPT = """
You are evaluating two images from the same FDM 3D-print layer. The first image is a phone side view. The second image is a top-down printer view.

Classify only the overall paired-image result as one of:
- normal
- under_extrusion
- unsure

Use severity 0 for normal, 1 for mild, 2 for moderate, and 3 for severe. If the evidence is weak, conflicting, blurry, occluded, or insufficient, use label "unsure".

Return strict JSON only:
{"label":"normal|under_extrusion|unsure","severity":0,"confidence":0.0,"evidence":"short visual evidence"}
""".strip()
