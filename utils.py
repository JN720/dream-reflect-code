def parse_code(code: str):
    try:
        code = code.strip()
        start = ''
        if '```python' in code.lower():
            start = code.lower().index("```python") + 9
        elif '```' in code.lower():
            start = code.lower().index("```") + 3
        else:
            start = 0
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

def parse_json_with_fallback(json_str: str):
    try:
        return True, parse_json(json_str)
    except:
        return False, json_str
