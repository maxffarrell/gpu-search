import re
import unicodedata

VERSION = 'gpu-search-features-v1'
WS = r'[\u0009-\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]'
def lower(text):
    return text.translate(str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'))
def normalize(text):
    return re.sub(WS + '+', ' ', lower(unicodedata.normalize('NFKC', text))).strip(' ')
def tokens(text):
    return [t for t in re.split('(?:' + WS + '|[_-])+', lower(re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', unicodedata.normalize('NFKC', text)))) if t]
def hash_bytes(data):
    h = 2166136261
    for byte in data:
        h = ((h ^ byte) * 16777619) & 0xffffffff
    return h & 1023
def features(text):
    all_tokens = tokens(text)
    ts = [t[:64] for t in all_tokens[:32]]
    words, chars = [], []
    total_chars = 0
    for t in ts:
        words.append(hash_bytes(('w:' + t).encode()))
        elems = [b'\x01'] + [b'\x00' + ord(c).to_bytes(4, 'little') for c in t] + [b'\x02']
        for n in (2, 3, 4):
            for i in range(len(elems)-n+1):
                total_chars += 1
                if len(chars) < 512:
                    chars.append(hash_bytes(b'c:' + b''.join(elems[i:i+n])))
    return {'wordIds': words, 'charIds': chars, 'wordCount': len(words), 'charCount': len(chars), 'truncated': len(all_tokens)>32 or any(len(t)>64 for t in all_tokens[:32]) or total_chars>512}
