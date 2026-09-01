import torch

from darwin_neg_router.neg import NEGEntropyMonitor, NEGStats


def monitor(entropy: torch.Tensor) -> NEGEntropyMonitor:
    value = NEGEntropyMonitor.__new__(NEGEntropyMonitor)
    value.threshold = 1.0
    value.temperature_scale = 0.5
    value.top_k = 2
    value.stats = NEGStats()
    value.last_entropy = entropy
    return value


def test_greedy_gate_records_but_does_not_pretend_to_rerank() -> None:
    value = monitor(torch.tensor([2.0]))
    scores = torch.tensor([[9.0, 8.0, 7.0]])
    output = value.apply_gate(scores, sampled=False)
    assert torch.equal(output, scores)
    assert value.stats.activations == 1


def test_sampled_gate_is_batch_safe() -> None:
    value = monitor(torch.tensor([2.0, 0.5]))
    scores = torch.tensor([[9.0, 8.0, 7.0], [4.0, 3.0, 2.0]])
    output = value.apply_gate(scores, sampled=True)
    assert torch.isneginf(output[0, 2])
    assert torch.equal(output[1], scores[1])
    assert not torch.isnan(output).any()

