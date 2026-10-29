import requests
from urllib.parse import quote


def translate_text(text, target_language):
    """
    Translate English text into the selected language.

    Uses MyMemory as the translation service.
    """

    if not text or not text.strip():
        return ""

    try:
        # MyMemory translation API
        url = (
            "https://api.mymemory.translated.net/get"
            f"?q={quote(text)}"
            f"&langpair=en|{target_language}"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        translated_text = data["responseData"]["translatedText"]

        if translated_text:
            return translated_text

        return "Translation failed. No translation received."

    except requests.exceptions.RequestException as e:
        return f"Translation service unavailable: {e}"

    except Exception as e:
        return f"Translation failed: {e}"