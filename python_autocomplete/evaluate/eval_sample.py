import torch

from labml import logger, lab, monit
from labml.logger import Text, Style
from python_autocomplete.evaluate import Predictor
from python_autocomplete.evaluate.beam_search import NextWordPredictionComplete
from python_autocomplete.evaluate.factory import get_predictor


def evaluate(predictor: Predictor, text: str):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    correct = 0
    i = 0
    key_strokes = 0

    while i + 1 < len(text):
        prefix = text[:i + 1]
        stripped, prompt = predictor.rstrip(prefix)
        rest = prefix[len(stripped):]
        prediction_complete = NextWordPredictionComplete(rest, 5)
        prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(-1)

        predictions = predictor.get_next_word(prompt, None, rest, [1.], prediction_complete, 5)
        predictions.sort(key=lambda x: -x[0])
        if predictions:
            next_token = predictions[0].text[len(rest):]
        else:
            next_token = ''

        if next_token and next_token == text[i + 1: i + 1 + len(next_token)]:
            correct += len(next_token)
            right = True
        else:
            next_token = text[i + 1]
            right = False

        for j, c in enumerate(next_token):
            if c == '\n':
                logger.log(logs)
                line_no += 1
                logs = [(f"{line_no: 4d}: ", Text.meta)]
            elif c == '\r':
                continue
            else:
                if right:
                    if j == 0:
                        logs.append((c, [Text.meta, Style.underline]))
                    else:
                        logs.append((c, [Text.success, Style.underline]))
                else:
                    logs.append((c, [Text.warning]))

        i += len(next_token)
        key_strokes += 1

    logger.log(logs)

    logger.inspect(accuracy=correct / (len(text) - 1),
                   key_strokes=key_strokes,
                   length=len(text))


def main():
    predictor = get_predictor()

    with open(str(lab.get_data_path() / 'sample.py'), 'r') as f:
        sample = f.read()
    with monit.section('Evaluate'):
        evaluate(predictor, sample)


if __name__ == '__main__':
    main()
