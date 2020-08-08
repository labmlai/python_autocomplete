import torch
import torch.nn
import torchtext
from labml import experiment, logger
from labml.helpers.pytorch.module import Module
from labml.logger import Text, Style
from labml.utils.pytorch import get_modules

from train import Configs


class Predictor:
    """
    Predicts the next few characters
    """

    def __init__(self, model: Module, field: torchtext.data.Field):
        self.field = field
        self.model = model

        # Initial state
        self._h0 = None
        self._c0 = None
        self._last_char = None

        # For timing
        self.time_add = 0
        self.time_predict = 0
        self.time_check = 0

    def get_predictions(self, char: str) -> torch.Tensor:
        data = torch.tensor([[self.field.vocab.stoi[char]]],
                            dtype=torch.long,
                            device=self.model.device)
        # Get predictions
        prediction, (h0, c0) = self.model(data, self._h0, self._c0)

        self._h0 = h0
        self._c0 = c0

        # Final prediction
        prediction = prediction[-1, :, :]

        return prediction.detach().cpu().numpy()

    def get_suggestion(self, char: str) -> str:
        prediction = self.get_predictions(char)
        best = prediction.argmax(-1).squeeze().item()
        return self.field.vocab.itos[best]


class Evaluator:
    def __init__(self, model: Module, field: torchtext.data.Field, text: str):
        self.text = text
        self.predictor = Predictor(model, field)

    def eval(self):
        line_no = 1
        logs = [(f"{line_no: 4d}: ", Text.meta), (self.text[0], Text.subtle)]

        correct = 0

        for i in range(len(self.text) - 1):
            next_char = self.predictor.get_suggestion(self.text[i])
            if next_char == self.text[i + 1]:
                correct += 1
            if self.text[i + 1] == '\n':
                logger.log(logs)
                line_no += 1
                logs = [(f"{line_no: 4d}: ", Text.meta)]
            elif self.text[i + 1] == '\r':
                continue
            else:
                if next_char == self.text[i + 1]:
                    logs.append((self.text[i + 1], Style.underline))
                else:
                    logs.append((self.text[i + 1], Text.subtle))

            # Log the line
        logger.log(logs)

        # Log time taken for the file
        logger.log("Accuracy: ", (f"{correct / (len(self.text) - 1) :.2f}", Text.value))


def main():
    conf = Configs()
    experiment.create(name="source_code",
                      comment='lstm model')

    conf_dict = experiment.load_configs('f940fc62d94611ea859dffe75b4de93a')
    conf_dict['n_tokens'] = 118
    experiment.configs(conf, conf_dict, 'run')
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load('f940fc62d94611ea859dffe75b4de93a')

    evaluator = Evaluator(conf.model, conf.field, conf.text.valid)
    evaluator.eval()


if __name__ == '__main__':
    main()
