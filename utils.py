def parse_code(code: str):
    try:
        code = code.strip()
        start = code.lower().index("```python") + 9
        end = code.index("```", start + 9)
        if start == -1 or end == -1:
            return code
        return code[start:end]
    except:
        return code

def parse_json(json_str: str):
    try:
        text = json_str.strip()
        start = text.lower().index("```json") + 7
        end = text.index("```", start + 7)
        if start == -1 or end == -1:
            return text
        return text[start:end]
    except:
        return text
