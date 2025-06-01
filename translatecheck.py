from googletrans import Translator

def translate_text():
    translator = Translator()
    text_to_translate = "Hello, how are you?"
    translated = translator.translate(text_to_translate, src='en', dest='hi')
    print(f"Translated text: {translated.text}")

translate_text()
