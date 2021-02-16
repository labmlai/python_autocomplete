import torch
from torch import nn

from labml import logger, lab, monit
from labml.logger import Text, Style
from labml_helpers.module import Module
from python_autocomplete.dataset import Tokenizer
from python_autocomplete.evaluate.factory import load_experiment
from python_autocomplete.train import StateUpdater


def anomalies(tokenizer: Tokenizer, text: str, model: Module, state_updater: StateUpdater, is_token_by_token: bool):
    tokens = tokenizer.encode(text)

    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (tokenizer.itos[tokens[0]], Style.bold)]

    text = torch.tensor(tokens, dtype=torch.long, device=model.device)
    prompt = text[:1].unsqueeze(-1)

    state = None
    softmax = nn.Softmax(-1)

    i = 1

    while i + 1 < len(text):
        with torch.no_grad():
            prediction, new_state = model(prompt, state)

        state = state_updater(state, new_state)
        prediction = softmax(prediction[-1, 0])

        if is_token_by_token:
            prompt = text[i: i + 1].unsqueeze(-1)
        else:
            prompt = text[:i + 1]
            prompt = prompt[-512:].unsqueeze(-1)

        token_str = tokenizer.itos[text[i]]
        prob = prediction[text[i]].item()

        for c in token_str:
            if c == '\n':
                logger.log(logs)
                line_no += 1
                logs = [(f"{line_no: 4d}: ", Text.meta)]
            elif c == '\r':
                continue
            else:
                if prob > 0.9:
                    logs.append((c, [Text.subtle, Style.underline]))
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
    conf = load_experiment()

    with open(str(lab.get_data_path() / 'sample.py'), 'r') as f:
        sample = f.read()
    with monit.section('Anomalies'):
        anomalies(conf.text.tokenizer, sample, conf.model, conf.state_updater, conf.is_token_by_token)


if __name__ == '__main__':
    main()
