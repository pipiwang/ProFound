import torch
import prettytable
import copy
import sys
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    get_scorer,
    get_scorer_names,
    make_scorer,
)


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100.0 / batch_size)
        return correct


def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, iu = self.compute()

        table = prettytable.PrettyTable(["class", "acc", "iou"])
        for i, class_name, per_acc, per_iu in zip(
            range(len(classes)), classes, (acc * 100).tolist(), (iu * 100).tolist()
        ):
            table.add_row([class_name, per_acc, per_iu])

        return (
            "global correct: {:.1f}\nmean correct:{:.1f}\nmean IoU: {:.1f}\n{}".format(
                acc_global.item() * 100,
                acc.mean().item() * 100,
                iu.mean().item() * 100,
                table.get_string(),
            )
        )


def kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[Union[str, np.ndarray]] = None,
    allow_off_by_one: bool = False,
) -> float:
    """
    Calculate the kappa inter-rater agreement.

    The agreement is calculated between the gold standard and the predicted
    ratings. Potential values range from -1 (representing complete disagreement)
    to 1 (representing complete agreement).  A kappa value of 0 is expected if
    all agreement is due to chance.

    In the course of calculating kappa, all items in ``y_true`` and ``y_pred`` will
    first be converted to floats and then rounded to integers.

    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.

    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true/actual/gold labels for the data.
    y_pred : numpy.ndarray
        The predicted/observed labels for the data.
    weights : Optional[Union[str, numpy.ndarray]], default=None
        Specifies the weight matrix for the calculation.
        Possible values are: ``None`` (unweighted-kappa), ``"quadratic"``
        (quadratically weighted kappa), ``"linear"`` (linearly weighted kappa),
        and a two-dimensional numpy array (a custom matrix of weights). Each
        weight in this array corresponds to the :math:`w_{ij}` values in the
        Wikipedia description of how to calculate weighted Cohen's kappa.
    allow_off_by_one : bool, default=False
        If true, ratings that are off by one are counted as
        equal, and all other differences are reduced by
        one. For example, 1 and 2 will be considered to be
        equal, whereas 1 and 3 will have a difference of 1
        for when building the weights matrix.

    Returns
    -------
    float
        The weighted or unweighted kappa score.

    Raises
    ------
    AssertionError
        If ``y_true`` != ``y_pred``.
    ValueError
        If labels cannot be converted to int.
    ValueError
        If invalid weight scheme.
    """
    # Ensure that the lists are both the same length
    assert len(y_true) == len(y_pred)

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    try:
        y_true = np.array([int(np.round(float(y))) for y in y_true])
        y_pred = np.array([int(np.round(float(y))) for y in y_pred])
    except ValueError:
        raise ValueError(
            "For kappa, the labels should be integers or strings"
            " that can be converted to ints (E.g., '4.0' or "
            "'3')."
        )

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = y_true - min_rating
    y_pred = y_pred - min_rating

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred, labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, str):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ""

    if weights is None:
        kappa_weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == "linear":
                    kappa_weights[i, j] = diff
                elif wt_scheme == "quadratic":
                    kappa_weights[i, j] = diff**2
                elif not wt_scheme:  # unweighted
                    kappa_weights[i, j] = bool(diff)
                else:
                    raise ValueError(
                        "Invalid weight scheme specified for " f"kappa: {wt_scheme}"
                    )
    else:
        kappa_weights = weights

    hist_true: np.ndarray = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / num_scored_items
    hist_pred: np.ndarray = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(kappa_weights):
        observed_sum = np.sum(kappa_weights * observed)
        expected_sum = np.sum(kappa_weights * expected)
        k -= np.sum(observed_sum) / np.sum(expected_sum)

    return k


def correlation(
    y_true: np.ndarray, y_pred: np.ndarray, corr_type: str = "pearson"
) -> float:
    """
    Calculate given correlation type between ``y_true`` and ``y_pred``.

    ``y_pred`` can be multi-dimensional. If ``y_pred`` is 1-dimensional, it
    may either contain probabilities, most-likely classification labels, or
    regressor predictions. In that case, we simply return the correlation
    between ``y_true`` and ``y_pred``. If ``y_pred`` is multi-dimensional,
    it contains probabilties for multiple classes in which case, we infer the
    most likely labels and then compute the correlation between those and
    ``y_true``.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true/actual/gold labels for the data.
    y_pred : numpy.ndarray
        The predicted/observed labels for the data.
    corr_type : str, default="pearson"
        Which type of correlation to compute. Possible
        choices are "pearson", "spearman", and "kendall_tau".

    Returns
    -------
    float
        correlation value if well-defined, else 0.0
    """
    # get the correlation function to use based on the given type
    corr_func = pearsonr
    if corr_type == "spearman":
        corr_func = spearmanr
    elif corr_type == "kendall_tau":
        corr_func = kendalltau

    # convert to numpy array in case we are passed a list
    y_pred = np.array(y_pred)

    # multi-dimensional -> probability array -> get label
    if y_pred.ndim > 1:
        labels = np.argmax(y_pred, axis=1)
        ret_score = corr_func(y_true, labels)[0]
    # 1-dimensional -> probabilities/labels -> use as is
    else:
        ret_score = corr_func(y_true, y_pred)[0]
    return ret_score


def f1_score_least_frequent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score of the least frequent label/class.

    Parameters
    ----------
    y_true : numpy.ndarray
        The true/actual/gold labels for the data.
    y_pred : numpy.ndarray
        The predicted/observed labels for the data.

    Returns
    -------
    float
        F1 score of the least frequent label.
    """
    least_frequent = np.bincount(y_true).argmin()
    return f1_score(y_true, y_pred, average=None)[least_frequent]
