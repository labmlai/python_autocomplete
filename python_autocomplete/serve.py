import json
import string
import threading

from flask import Flask, request, jsonify

from labml import monit
from python_autocomplete.evaluate import get_predictor

TOKEN_CHARS = set(string.ascii_letters + string.digits + ' ' + '\n' + '\r' + '_')

app = Flask('python_autocomplete')
predictor = get_predictor()
lock = threading.Lock()


@app.route('/')
def home():
    return 'Home'


@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    prompt = request.json['prompt']
    if not prompt:
        return jsonify({'success': False})

    with monit.section('Predict') as s:
        acquired = lock.acquire(blocking=False)
        if acquired:
            res = predictor.get_token(prompt, token_chars=TOKEN_CHARS)
            lock.release()
            s.message = f'{json.dumps(prompt[-5:])} -> {json.dumps(res)}'
            return jsonify({'success': True, 'prediction': res})
        else:
            monit.fail()
            return jsonify({'success': False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
