"""Flask web server for NLP Emotion Detection application."""

from flask import Flask, render_template, request, jsonify
from emotion_detection import emotion_detector


app = Flask(__name__)


@app.route('/')
def index():
    """Render the main index.html template."""
    return render_template('index.html')


@app.route('/textanalyze', methods=['POST'])
def emotion_detector_endpoint():
    """Analyze text emotion using Watson NLP and return results as JSON.
    
    Expects POST request with JSON body containing 'text' field.
    Returns emotion scores and dominant emotion, or error message for invalid input.
    
    Returns:
        jsonify: JSON response with emotion analysis results or error message.
    """
    data = request.get_json()
    text_to_analyze = data.get('text', '').strip()
    
    result = emotion_detector(text_to_analyze)
    
    if result['dominant_emotion'] is None:
        return jsonify({
            'label': 'Invalid text! Please try again!',
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        })
    
    response_text = (
        f"'anger': {result['anger']:.7g}, "
        f"'disgust': {result['disgust']:.7g}, "
        f"'fear': {result['fear']:.7g}, "
        f"'joy': {result['joy']:.7g} "
        f"and 'sadness': {result['sadness']:.7g}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )
    
    return jsonify({
        'label': response_text,
        'anger': result['anger'],
        'disgust': result['disgust'],
        'fear': result['fear'],
        'joy': result['joy'],
        'sadness': result['sadness'],
        'dominant_emotion': result['dominant_emotion']
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
