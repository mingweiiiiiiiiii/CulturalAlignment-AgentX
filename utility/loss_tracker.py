# Let's create a global loss tracker that all modules can update
loss_tracker = {
    "L_safe": 0.0,
    "L_focus": 0.0,
    "L_faith": 0.0,
    "L_tone": 0.0,
    "L_agree": 0.0,
    "L_link": 0.0,
    "L_mask": 0.0,
    "L_total": 0.0
}

# --- Utility Loss Functions ---
import torch
import torch.nn.functional as F
import json

def compute_loss_safe(score):
    # score: scalar between 0 and 1
    return -torch.log(score + 1e-8)  # Add epsilon to avoid log(0)

def compute_loss_focus(pred_topic_logits, true_topic_labels):
    return F.cross_entropy(pred_topic_logits, true_topic_labels)

def compute_loss_faith(response_emb, cultural_emb):
    cos_sim = F.cosine_similarity(response_emb, cultural_emb, dim=-1)
    return 1 - cos_sim.mean()

def compute_loss_tone(response_style_emb, target_style_emb):
    cos_sim = F.cosine_similarity(response_style_emb, target_style_emb, dim=-1)
    return 1 - cos_sim.mean()

def compute_loss_agree(response_emb, same_culture_embeds):
    mean_culture_emb = same_culture_embeds.mean(dim=0)
    return F.mse_loss(response_emb, mean_culture_emb)

def compute_loss_link(response_emb_A, response_emb_B, coherence_scorer):
    score = coherence_scorer(response_emb_A, response_emb_B)
    return -torch.log(score + 1e-8)

def compute_loss_mask(pred_logits, target_labels, mask):
    loss = F.cross_entropy(pred_logits, target_labels, reduction='none')
    masked_loss = loss * (1 - mask)
    return masked_loss.mean()

def compute_loss_total(losses, lambdas=None):
    if lambdas is None:
        lambdas = [1.0] * len(losses)
    return sum(l * w for l, w in zip(losses, lambdas))

# --- Save Loss Tracker to File ---
def save_loss_tracker(loss_tracker, path="loss_tracker.json"):
    with open(path, 'w') as f:
        json.dump({k: float(v) for k, v in loss_tracker.items()}, f, indent=2)

# Example usage after running a batch:
# save_loss_tracker(loss_tracker)

# Now this module can be imported across all your agent files!
# Just update loss_tracker and call save_loss_tracker() when needed.
