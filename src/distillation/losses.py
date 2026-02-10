"""
MaxSim Interaction Distillation (MID) loss components.

Implements the three-part loss from Idea 1:
  L_total = α · L_contrastive + β · L_interaction + γ · L_ranking

Where:
  - L_contrastive: Standard query-doc contrastive loss
  - L_interaction: KL-divergence between teacher/student MaxSim attention maps
  - L_ranking: Margin ranking loss on teacher vs student score orderings
"""

import torch
import torch.nn.functional as F


def contrastive_loss(
    query_embeddings: torch.Tensor,
    pos_doc_embeddings: torch.Tensor,
    neg_doc_embeddings: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Standard contrastive loss using MaxSim scoring.

    Args:
        query_embeddings: (B, Tq, D) query token embeddings
        pos_doc_embeddings: (B, Td, D) positive document token embeddings
        neg_doc_embeddings: (B, N, Td, D) negative document token embeddings
        temperature: softmax temperature

    Returns:
        Scalar contrastive loss
    """
    # MaxSim scores for positive pairs
    pos_scores = _maxsim_score(query_embeddings, pos_doc_embeddings)  # (B,)

    # MaxSim scores for negative pairs
    B, N, Td, D = neg_doc_embeddings.shape
    neg_flat = neg_doc_embeddings.view(B * N, Td, D)
    query_expanded = query_embeddings.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, -1, D)
    neg_scores = _maxsim_score(query_expanded, neg_flat).view(B, N)  # (B, N)

    # InfoNCE loss
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) / temperature
    labels = torch.zeros(B, dtype=torch.long, device=all_scores.device)
    return F.cross_entropy(all_scores, labels)


def interaction_loss(
    student_query_emb: torch.Tensor,
    student_doc_emb: torch.Tensor,
    teacher_attention: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL-divergence between teacher and student MaxSim attention maps.

    This distills WHICH query token should match WHICH doc token.

    Args:
        student_query_emb: (B, Tq, D) student query embeddings
        student_doc_emb: (B, Td, D) student doc embeddings
        teacher_attention: (B, Tq, Td) teacher's softmax attention maps
        temperature: temperature for softening distributions

    Returns:
        Scalar KL-divergence loss
    """
    # Student's raw similarity scores
    student_sim = torch.bmm(student_query_emb, student_doc_emb.transpose(1, 2))  # (B, Tq, Td)
    student_log_probs = F.log_softmax(student_sim / temperature, dim=-1)

    # Teacher's softened targets
    teacher_probs = F.softmax(
        torch.log(teacher_attention + 1e-8) / temperature, dim=-1
    )

    # KL divergence
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kl * (temperature ** 2)  # Scale by T^2 as in standard KD


def ranking_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Margin ranking loss to align student and teacher score orderings.

    For each pair of documents, if teacher ranks doc_i > doc_j,
    then student should also rank doc_i > doc_j with at least `margin`.

    Args:
        student_scores: (B, N) student MaxSim scores for N documents
        teacher_scores: (B, N) teacher MaxSim scores for N documents
        margin: minimum margin for correct ordering

    Returns:
        Scalar ranking loss
    """
    B, N = student_scores.shape

    # Create all pairs (i, j) where teacher_score[i] > teacher_score[j]
    s_i = student_scores.unsqueeze(2).expand(-1, -1, N)  # (B, N, N)
    s_j = student_scores.unsqueeze(1).expand(-1, N, -1)  # (B, N, N)

    t_i = teacher_scores.unsqueeze(2).expand(-1, -1, N)
    t_j = teacher_scores.unsqueeze(1).expand(-1, N, -1)

    # Mask: only consider pairs where teacher has a clear preference
    teacher_prefers_i = (t_i > t_j).float()

    # Hinge loss: max(0, margin - (s_i - s_j)) where teacher prefers i
    pairwise_loss = F.relu(margin - (s_i - s_j)) * teacher_prefers_i

    # Average over valid pairs
    num_pairs = teacher_prefers_i.sum().clamp(min=1)
    return pairwise_loss.sum() / num_pairs


def mid_loss(
    l_contrastive: torch.Tensor,
    l_interaction: torch.Tensor,
    l_ranking: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Combined MaxSim Interaction Distillation (MID) loss.

    L_total = α · L_contrastive + β · L_interaction + γ · L_ranking
    """
    return alpha * l_contrastive + beta * l_interaction + gamma * l_ranking


def _maxsim_score(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute MaxSim score: Σ_q max_d (q · d).

    Args:
        query_embeddings: (B, Tq, D)
        doc_embeddings: (B, Td, D)

    Returns:
        scores: (B,) MaxSim scores
    """
    # (B, Tq, Td) similarity matrix
    sim = torch.bmm(query_embeddings, doc_embeddings.transpose(1, 2))
    # Max over doc tokens for each query token, then sum
    max_sim = sim.max(dim=-1).values  # (B, Tq)
    return max_sim.sum(dim=-1)  # (B,)
