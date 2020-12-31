import threading

from flask import Flask, request, jsonify

from labml import experiment
from labml.utils.cache import cache
from labml.utils.pytorch import get_modules
from python_autocomplete.evaluate import Predictor
from python_autocomplete.train import Configs


def get_predictor():
    conf = Configs()
    experiment.evaluate()

    # Replace this with your training experiment UUID
    run_uuid = '39b03a1e454011ebbaff2b26e3148b3d'

    conf_dict = experiment.load_configs(run_uuid)
    experiment.configs(conf, conf_dict)
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load(run_uuid)

    experiment.start()
    conf.model.eval()
    return Predictor(conf.model, cache('stoi', lambda: conf.text.stoi), cache('itos', lambda: conf.text.itos))


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

    acquired = lock.acquire(blocking=False)
    if acquired:
        res = predictor.get_token(prompt)
        lock.release()
        return jsonify({'success': True, 'prediction': res})
    else:
        return jsonify({'success': False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
