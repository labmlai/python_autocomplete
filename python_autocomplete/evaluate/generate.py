from labml import logger, lab, monit
from labml.logger import Text, Style
from python_autocomplete.evaluate import Predictor
from python_autocomplete.evaluate.factory import get_predictor


def generate(predictor: Predictor, text: str, completion: int):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    i = 0
    given = len(text)

    while i + 1 < given + completion:
        if len(text) > i + 1:
            c = text[i + 1]
        else:
            c, _ = predictor.get_next_token(text[:i + 1], None)

        if c == '\n':
            logger.log(logs)
            line_no += 1
            logs = [(f"{line_no: 4d}: ", Text.meta)]
        elif c != '\r':
            if len(text) > i + 1:
                logs.append(c)
            else:
                logs.append((c, [Style.bold]))

        if len(text) <= i + 1:
            text += c

        i += 1

    logger.log(logs)


def main():
    predictor = get_predictor()

    with open(str(lab.get_data_path() / 'sample.py'), 'r') as f:
        sample = f.read()
    with monit.section('Generate'):
        generate(predictor, 'import numpy as np\n', 1000)


if __name__ == '__main__':
    main()
