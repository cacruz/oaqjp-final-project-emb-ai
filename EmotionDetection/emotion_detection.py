import requests
import json


def emotion_detector(text_to_analyze: str) -> dict:
    """
    Returns:
    {
      'anger': <float or None>,
      'disgust': <float or None>,
      'fear': <float or None>,
      'joy': <float or None>,
      'sadness': <float or None>,
      'dominant_emotion': <str or None>
    }
    """
    # Required "all None" response format
    none_result = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }

    # Handle blank / missing input from user
    if text_to_analyze is None or not text_to_analyze.strip():
        return none_result

    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    payload = {"raw_document": {"text": text_to_analyze}}
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    response = requests.post(url, json=payload, headers=headers)

    # Access status_code to manage blank/invalid entries (per requirement)
    if response.status_code == 400:
        return none_result

    # Optional: if other non-200 errors occur, fail gracefully
    if response.status_code != 200:
        return none_result

    # Convert response text into a dictionary
    try:
        response_dict = json.loads(response.text)
    except json.JSONDecodeError:
        return none_result

    # Extract emotions safely
    try:
        emotions_src = response_dict["emotionPredictions"][0]["emotion"]
        emotions = {
            "anger": emotions_src.get("anger"),
            "disgust": emotions_src.get("disgust"),
            "fear": emotions_src.get("fear"),
            "joy": emotions_src.get("joy"),
            "sadness": emotions_src.get("sadness"),
        }
    except (KeyError, IndexError, TypeError):
        return none_result

    # Find dominant emotion (highest score)
    dominant_emotion = max(emotions, key=emotions.get) if emotions else None

    return {**emotions, "dominant_emotion": dominant_emotion}
