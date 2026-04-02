def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    """
    Encrypts plaintext using a Vigenere cipher.

    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    ciphertext = ""
    keyword = keyword.lower()
    key_index = 0

    for char in plaintext:
        if char.isupper():
            shift = ord(keyword[key_index % len(keyword)]) - ord('a')
            
            position = ord(char) - ord('A')
            new_position = (position + shift) % 26
            new_char = chr(new_position + ord('A'))
            ciphertext += new_char
            
            key_index += 1
            
        elif char.islower():
            shift = ord(keyword[key_index % len(keyword)]) - ord('a')
            
            position = ord(char) - ord('a')
            new_position = (position + shift) % 26
            new_char = chr(new_position + ord('a'))
            ciphertext += new_char
            
            key_index += 1
            
        else:
            ciphertext += char
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    keyword = keyword.lower()
    key_index = 0
    
    for char in ciphertext:
        if char.isupper():
            shift = ord(keyword[key_index % len(keyword)]) - ord('a')
            
            position = ord(char) - ord('A')
            new_position = (position - shift) % 26
            new_char = chr(new_position + ord('A'))
            plaintext += new_char
            
            key_index += 1
            
        elif char.islower():
            shift = ord(keyword[key_index % len(keyword)]) - ord('a')
            
            position = ord(char) - ord('a')
            new_position = (position - shift) % 26
            new_char = chr(new_position + ord('a'))
            plaintext += new_char
            
            key_index += 1
            
        else:
            plaintext += char
    return plaintext
