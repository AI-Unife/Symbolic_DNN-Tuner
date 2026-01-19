import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings

from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class RegressionUsability(Metric):
    def __init__(
        self,
        minimum_error: float,
        maximum_error: float,
        tolerance: float,
        output_transform=lambda x: x,
        device="cpu",
    ) -> None:
        self._num_correct = None
        self._num_examples = None
        self.minimum_diff = minimum_error
        self.maximum_diff = maximum_error
        self.tolerance = tolerance
        super(RegressionUsability, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y_true = output[0].detach(), output[1].detach()

        y_diff = torch.absolute(y_pred - y_true)
        correct = (y_diff < self.minimum_diff) | (
            ~(y_diff > self.maximum_diff)
            & (torch.absolute(y_pred - y_true / y_true) < self.tolerance)
        )
        for i in range(y_pred.shape[0]):
            print(f"{y_pred[i]} {y_true[i]} {correct[i]}")
        self._num_correct += correct.sum().to(self._device)
        self._num_examples += y_true.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if self._num_examples is None or self._num_examples == 0 or self._num_correct is None:
            raise NotComputableError(
                "RegressionUsability must have at least one example before it can be computed."
            )
        return self._num_correct.item() / self._num_examples
