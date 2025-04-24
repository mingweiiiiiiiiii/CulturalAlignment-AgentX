# rtsso
# Load BERT model and tokenizer for scoring and embedding
from transformers import  AutoTokenizer, AutoModel
import torch
# import matplotlib.pyplot as plt

# Lad BER model and ter for scoring and embedding
class BERTWrapper:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

    def softmax_classify(self, text, label_map):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]
            logits = torch.nn.Linear(cls_token.shape[-1], len(label_map))(cls_token)
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0)

class EvaluationLosses:
    def __init__(self, lambdas, label_map):
        self.lambdas = lambdas
        self.label_map = label_map
        self.bert = BERTWrapper()

    def L_safe(self, response):
        """Calculates the sensitivity filtering loss."""
        probs = self.bert.softmax_classify(response, self.label_map)
        safe_idx = self.label_map["safe"]
        return -torch.log(probs[safe_idx] + 1e-8)

    def L_focus(self, responses, topics):
        """Calculates the topic grounding loss."""
        loss = 0.0
        for r, t in zip(responses, topics):
            probs = self.bert.softmax_classify(r, self.label_map)
            target_idx = self.label_map[t]
            loss += F.cross_entropy(probs.unsqueeze(0), torch.tensor([target_idx]))
        return loss / len(responses)

    def L_faith(self, response, cultural_ref):
        """Calculates the cultural fidelity loss."""
        emb_r = self.bert.embed(response).squeeze(0)
        emb_c = self.bert.embed(cultural_ref).squeeze(0)
        cos_sim = F.cosine_similarity(emb_r, emb_c, dim=0)
        return 1 - cos_sim

    def L_tone(self, response, target_style):
        """Calculates the stylistic consistency loss."""
        style_vec = self.bert.embed(response).squeeze(0)
        cos_sim = F.cosine_similarity(style_vec, target_style, dim=0)
        return 1 - cos_sim

    def L_agree(self, response, cultural_group_responses):
        """Calculates the intra-cultural agreement loss."""
        emb_r = self.bert.embed(response).squeeze(0)
        group_embeds = torch.stack([self.bert.embed(r).squeeze(0) for r in cultural_group_responses])
        mu_c = group_embeds.mean(dim=0)
        return torch.norm(emb_r - mu_c, p=2)

    def L_link(self, responseA, responseB):
        """Calculates the cross-cultural coherence loss."""
        emb_A = self.bert.embed(responseA).squeeze(0)
        emb_B = self.bert.embed(responseB).squeeze(0)
        return -torch.log(F.cosine_similarity(emb_A, emb_B, dim=0) + 1e-8)

    def L_mask(self, predictions, labels, masks):
        """Calculates the prompt bias reduction loss."""
        loss = 0.0
        for i in range(len(predictions)):
            if masks[i] == 1:
                loss += F.cross_entropy(predictions[i].unsqueeze(0), labels[i].unsqueeze(0))
        return loss / (masks.sum() + 1e-8)

    def L_total(self, response_pack):
        """Calculates the composite system loss."""
        losses = [
            self.L_safe(response_pack['response']),
            self.L_focus(response_pack['topic_responses'], response_pack['topics']),
            self.L_faith(response_pack['response'], response_pack['cultural_ref']),
            self.L_tone(response_pack['response'], response_pack['style']),
            self.L_agree(response_pack['response'], response_pack['same_culture_responses']),
            self.L_link(response_pack['responseA'], response_pack['responseB']),
            self.L_mask(response_pack['predictions'], response_pack['labels'], response_pack['masks'])
        ]
        total = sum(self.lambdas[i] * losses[i] for i in range(len(losses)))
        return total

# def save_evaluation_chart(metrics):
#     """Generates and saves a chart visualizing the evaluation metrics."""
#     plt.figure(figsize=(10, 5))
#     plt.plot(metrics, marker='o', label='Total Loss')
#     plt.title('Evaluation Metrics Over Time')
#     plt.xlabel('Evaluation Iteration')
#     plt.ylabel('Loss Value')
#     plt.legend()
#     plt.grid()
#     plt.savefig('evaluation_metrics_chart.png')
#     plt.close()
