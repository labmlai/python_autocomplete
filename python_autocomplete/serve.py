import json
import threading

import torch
from flask import Flask, request, jsonify

from labml import monit
from python_autocomplete.evaluate.beam_search import NextWordPredictionComplete
from python_autocomplete.evaluate.factory import get_predictor

app = Flask('python_autocomplete')
predictor = get_predictor()
lock = threading.Lock()


@app.route('/')
def home():
    return 'Home'


@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    prefix = request.json['prompt']
    if not prefix:
        return jsonify({'success': False})

    with monit.section('Predict') as s:
        acquired = lock.acquire(blocking=False)
        if acquired:
            stripped, prompt = predictor.rstrip(prefix)
            rest = prefix[len(stripped):]
            prediction_complete = NextWordPredictionComplete(rest, 15)
            prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(-1)

            predictions = predictor.get_next_word(prompt, None, rest, [1.], prediction_complete, 5)
            predictions.sort(key=lambda x: -x.prob)

            results = [pred.text[len(rest):] for pred in predictions]
            probs = [pred.prob for pred in predictions]
            lock.release()
            s.message = f'{json.dumps(prefix[-5:])} -> {json.dumps(results)}'
            return jsonify({'success': True, 'prediction': results, 'probs': probs})
        else:
            monit.fail()
            return jsonify({'success': False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
