from labml import logger, lab, monit
from labml.logger import Text, Style
from python_autocomplete.evaluate import Predictor
from python_autocomplete.evaluate.factory import get_predictor


def anomalies(predictor: Predictor, text: str):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    i = 0

    while i + 1 < len(text):
        #             print(i, self.predictor.prompt)
        preds, _ = predictor.get_predictions(text[:i + 1], None, calc_probs=True)
        preds = preds[0, :]
        c = text[i + 1]

        if c == '\n':
            logger.log(logs)
            line_no += 1
            logs = [(f"{line_no: 4d}: ", Text.meta)]
        elif c == '\r':
            continue
        elif c not in predictor.tokenizer.stoi:
            logs.append(c)
        else:
            next_id = predictor.tokenizer.stoi[c]
            prob = preds[next_id]
            if prob > 0.9:
                logs.append((c, [Style.bold, Text.success, Style.underline]))
            elif prob > 0.75:
                logs.append((c, [Text.success, Style.underline]))
            elif prob > 0.2:
                logs.append(c)
            elif prob > 0.1:
                logs.append((c, [Text.warning, Style.underline]))
            elif prob > 0.01:
                logs.append((c, [Style.bold, Text.warning, Style.underline]))
            elif prob > 0.001:
                logs.append((c, [Text.danger, Style.underline]))
            else:
                logs.append((c, [Style.bold, Text.danger, Style.underline]))

        i += 1

    logger.log(logs)


def main():
    predictor = get_predictor()

    with open(str(lab.get_data_path() / 'sample.py'), 'r') as f:
        sample = f.read()
    with monit.section('Anomalies'):
        anomalies(predictor, sample)


if __name__ == '__main__':
    main()
