from deep_translator import GoogleTranslator

def translate_word(word, lang):
    try:
        return GoogleTranslator(source='auto', target=lang).translate(word).lower()
    except Exception as e:
        print(f"Error translating {word} to {lang}: {e}")
        return word

def translate_words(words, lang):
    print(f"Translating to {lang}...")
    return [translate_word(w, lang) for w in words]
