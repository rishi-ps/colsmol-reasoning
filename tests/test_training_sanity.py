import torch

from src.models.teacher import TeacherModelWrapper
from src.reasoning.trainer import R2RTrainer, R2RTrainerConfig


def test_r2r_compute_loss_requires_in_batch_negatives():
    trainer = R2RTrainer(retriever=None, config=R2RTrainerConfig())
    q = torch.randn(1, 4, 8)
    d = torch.randn(1, 6, 8)
    try:
        trainer.compute_loss(q, d)
        assert False, "Expected ValueError for batch_size < 2"
    except ValueError:
        pass


def test_r2r_compute_loss_runs_for_batch_size_two():
    trainer = R2RTrainer(retriever=None, config=R2RTrainerConfig())
    q = torch.randn(2, 4, 8)
    d = torch.randn(2, 6, 8)
    loss = trainer.compute_loss(q, d)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_teacher_attention_shape_matches_batch_token_dims():
    teacher = TeacherModelWrapper()
    q = torch.randn(2, 5, 16)
    d = torch.randn(2, 7, 16)
    attn = teacher.get_maxsim_attention(q, d)
    assert attn.shape == (2, 5, 7)
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
