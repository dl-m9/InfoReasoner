"""Implement semantic entropy."""
import os
import pickle
import logging

import numpy as np
import wandb
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..models.huggingface_models import HuggingfaceModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ENTAILMENT_THRESHOLD = 0.5


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self, device=None):
        self.device = device if device is not None else DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(self.device)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(
            text1, 
            text2,
            padding=True, 
            truncation=True,
            return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1), dim=1)  # pylint: disable=no-member
        prediction = largest_index.detach()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction, logits.detach()



def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None, threshold=DEFAULT_ENTAILMENT_THRESHOLD):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):

        implication_1, logits_1 = model.check_implication(text1, text2, example=example)
        implication_2, logits_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        implication_1 = implication_1.item() if torch.is_tensor(implication_1) else implication_1
        implication_2 = implication_2.item() if torch.is_tensor(implication_2) else implication_2
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            entail_prob_1 = F.softmax(logits_1, dim=-1)[..., 2].item()
            entail_prob_2 = F.softmax(logits_2, dim=-1)[..., 2].item()
            semantically_equivalent = (entail_prob_1 > threshold) and (entail_prob_2 > threshold)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    adjacency = [[] for _ in range(len(strings_list))]
    for i, string1 in enumerate(strings_list):
        for j in range(i + 1, len(strings_list)):
            if are_equivalent(string1, strings_list[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for start in range(len(strings_list)):
        if semantic_set_ids[start] != -1:
            continue
        stack = [start]
        semantic_set_ids[start] = next_id
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if semantic_set_ids[neighbor] == -1:
                    semantic_set_ids[neighbor] = next_id
                    stack.append(neighbor)
        next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy
