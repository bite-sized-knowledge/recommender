import re

# \xNN 형태를 실제 문자로 변환
def replace_x(match):
        hex_value = match.group(1)
        try:
            return chr(int(hex_value, 16))
        except:
            return match.group(0)

 #\uXXXX 형태를 실제 문자로 변환
def replace_u(match):
    hex_value = match.group(1)
    try:
        return chr(int(hex_value, 16))
    except:
        return match.group(0)

def decode_unicode_escapes(s):
    s = re.sub(r'\\u([0-9a-fA-F]{4})', replace_u, s)
    s = re.sub(r'\\x([0-9a-fA-F]{2})', replace_x, s)
    return s