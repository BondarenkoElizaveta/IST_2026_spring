import typing as tp


def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """
    ciphertext = ""
    for char in plaintext:
        if char.isupper():
            position = ord(char) - ord('A')
            new_position = (position + shift)%26
            new_char = chr(new_position + ord('A'))
            ciphertext += new_char
        elif char.islower():
            position = ord(char) - ord('a')
            new_position = (position + shift) % 26
            new_char = chr(new_position + ord('a'))
            ciphertext += new_char
        else:
            ciphertext += char
    return ciphertext


def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    plaintext = ""
    for char in ciphertext:
        if char.isupper():
            position = ord(char) - ord('A')
            new_position = (position - shift)%26
            new_char = chr(new_position + ord('A'))
            plaintext += new_char
        elif char.islower():
            position = ord(char) - ord('a')
            new_position = (position - shift) % 26
            new_char = chr(new_position + ord('a'))
            plaintext += new_char
        else:
            plaintext += char
    return plaintext


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0
    max_matches = 0
    for shift in range(1, 26):
        decrypted = decrypt_caesar(ciphertext, shift)
        words = decrypted.split()
        
        matches = 0
        for word in words:
            clean_word = word.strip(".,!?;:()[]{}'\"")
            clean_word = clean_word.lower()
            
            if clean_word in dictionary:
                matches += 1
        
        if matches > max_matches:
            max_matches = matches
            best_shift = shift
    return best_shift
