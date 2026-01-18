import requests
import json


def emotion_detector(text_to_analyze: str) -> dict:
    """
    Calls the Watson EmotionPredict endpoint and returns a simplified dict of
    emotion scores + dominant emotion.

    Output format:
    {
      'anger': <float>,
      'disgust': <float>,
      'fear': <float>,
      'joy': <float>,
      'sadness': <float>,
      'dominant_emotion': <str or None>
    }
    """
    url = (
        "https://sn-watson-emotion.labs.skills.network/v1/"
        "watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    payload = {"raw_document": {"text": text_to_analyze}}
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    response = requests.post(url, json=payload, headers=headers)

    # Optional but recommended: fail fast on HTTP errors
    response.raise_for_status()

    # Convert response text into a dictionary
    response_dict = json.loads(response.text)

    # Extract required emotions + scores
    emotions_src = response_dict["emotionPredictions"][0]["emotion"]

    anger = emotions_src.get("anger", 0.0)
    disgust = emotions_src.get("disgust", 0.0)
    fear = emotions_src.get("fear", 0.0)
    joy = emotions_src.get("joy", 0.0)
    sadness = emotions_src.get("sadness", 0.0)

    emotions = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
    }

    # Find dominant emotion (highest score)
    dominant_emotion = max(emotions, key=emotions.get) if emotions else None

    # Return in the required format
    return {
        **emotions,
        "dominant_emotion": dominant_emotion,
    }
