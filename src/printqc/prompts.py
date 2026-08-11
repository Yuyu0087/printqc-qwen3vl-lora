PROMPT_ID = "printqc-paired-classification-zh-v1"

PAIRED_CLASSIFICATION_PROMPT = """
这是同一个3D打印件同一层的两张图:第一张是手机侧拍高清图,第二张是机载俯拍图。请综合两个视角判断该层是否存在欠挤出(under-extrusion)缺陷。输出整件类别 normal/under_extrusion/unsure 与严重度 severity(0无/1轻/2中/3重),并简述依据。
""".strip()
