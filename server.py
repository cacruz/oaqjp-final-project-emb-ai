"""
server.py

This module implements a Flask web application that exposes an emotion
detection service using a pre-trained NLP emotion model. The application
provides a web interface and an HTTP endpoint for analyzing user-submitted
text and identifying the dominant emotion expressed in the text.
"""


from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")

@app.route("/emotionDetector")
def sent_analyzer():
    """
    Analyzes user-provided text and determines the dominant emotion expressed.

    Returns
    -------
    str
        A formatted message describing the dominant emotion and its score,
        or an error message if the input text is invalid.
    """
    text_to_analyze = request.args.get('textToAnalyze')

    response = emotion_detector(text_to_analyze)

    label = response.get('dominant_emotion')

    # Error handling: dominant_emotion is None
    if label is None:
        return "Invalid text! Please try again!"

    score = response.get(label)

    return f"The given text has been identified as {label} with a score of {score}."

@app.route("/")
def render_index_page():
    ''' This function initiates the rendering of the main application
        page over the Flask channel
    '''
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
