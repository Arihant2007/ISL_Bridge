from deep_translator import GoogleTranslator

def translate(sentence,language):
    translator = GoogleTranslator(source = "en", target = language)
    translated_sentence = translator.translate(sentence)
    return translated_sentence
