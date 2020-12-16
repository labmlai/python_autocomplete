import torch
import torch.nn
from labml import experiment, logger
from labml_helpers.module import Module
from labml.logger import Text, Style
from labml.utils.pytorch import get_modules

from python_autocomplete.train import Configs, TextDataset


class Predictor:
    """
    Predicts the next few characters
    """

    def __init__(self, model: Module, dataset: TextDataset, is_lstm=True):
        self.is_lstm = is_lstm
        self.dataset = dataset
        self.model = model

        # Initial state
        self._h0 = None
        self._c0 = None
        self.prompt = ''

        # For timing
        self.time_add = 0
        self.time_predict = 0
        self.time_check = 0

    def get_predictions(self, char: str) -> torch.Tensor:
        if self.is_lstm:
            return self.get_predictions_lstm(char)
        else:
            return self.get_predictions_transformer(char)

    def get_predictions_transformer(self, char: str) -> torch.Tensor:
        self.prompt += char
        self.prompt = self.prompt[-512:]
        data = torch.tensor([[self.dataset.stoi[c]] for c in self.prompt],
                            dtype=torch.long,
                            device=self.model.device)

        # Get predictions
        prediction, *_ = self.model(data)

        # Final prediction
        prediction = prediction[-1, :, :]

        return prediction.detach().cpu().numpy()

    def get_predictions_lstm(self, char: str) -> torch.Tensor:
        data = torch.tensor([[self.dataset.stoi[char]]],
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
        return self.dataset.itos[best]


class Evaluator:
    def __init__(self, model: Module, dataset: TextDataset, text: str, is_lstm=True):
        self.text = text
        self.predictor = Predictor(model, dataset, is_lstm)

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
                    logs.append((self.text[i + 1], [Text.success, Style.underline]))
                else:
                    logs.append((self.text[i + 1], Text.subtle))

            # Log the line
        logger.log(logs)

        # Log time taken for the file
        logger.log("Accuracy: ", (f"{correct / (len(self.text) - 1) :.2f}", Text.value))


def main():
    conf = Configs()
    experiment.evaluate()

    # Replace this with your training experiment UUID
    conf_dict = experiment.load_configs('8d16abcc3f6211ebb0be67ed81588441')
    experiment.configs(conf, conf_dict)
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load('8d16abcc3f6211ebb0be67ed81588441')

    experiment.start()
    from python_autocomplete.models.transformer import TransformerModel
    evaluator = Evaluator(conf.model, conf.text, conf.text.valid, not isinstance(conf.model, TransformerModel))
    evaluator.eval()


if __name__ == '__main__':
    main()
