from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
from llm_prep.utils.testing import load_local_starter, skip_not_implemented
starter = load_local_starter(Path(__file__).with_name("starter.py"))

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
@pytest.mark.top10
def test_mini_transformer_lm_returns_vocab_logits() -> None:
    try:
        model = starter.MiniTransformerLM(vocab_size=32, hidden_size=16, num_heads=4, intermediate_size=32)
        logits = model(torch.randint(0, 32, (2, 5)))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(logits.shape) == (2, 5, 32)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_mini_transformer_lm_single_token() -> None:
    """单 token 输入应正常"""
    try:
        model = starter.MiniTransformerLM(vocab_size=32, hidden_size=16, num_heads=4, intermediate_size=32)
        logits = model(torch.randint(0, 32, (1, 1)))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(logits.shape) == (1, 1, 32)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_mini_transformer_lm_different_vocab_size() -> None:
    """不同 vocab_size 应正确映射"""
    try:
        model = starter.MiniTransformerLM(vocab_size=100, hidden_size=16, num_heads=4, intermediate_size=32)
        logits = model(torch.randint(0, 100, (1, 3)))
    except Exception as exc:
        skip_not_implemented(exc)
    assert tuple(logits.shape) == (1, 3, 100)

@pytest.mark.practice
@pytest.mark.transformer
@pytest.mark.difficulty_4
def test_mini_transformer_lm_gradient_flows() -> None:
    """梯度应该能正常回传"""
    try:
        model = starter.MiniTransformerLM(vocab_size=32, hidden_size=16, num_heads=4, intermediate_size=32)
        logits = model(torch.randint(0, 32, (1, 3)))
        loss = logits.sum()
        loss.backward()
    except Exception as exc:
        skip_not_implemented(exc)
    # 如果没报错就说明梯度正常回传
    assert model.token_emb.weight.grad is not None
