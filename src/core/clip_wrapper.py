"""CLIP ViT-L/14 wrapper: joint-space embeddings, token vectors, similarities.

Joint space is 768-d (projected CLS / EOS pooled vectors) - use it for all
image<->text similarity and vector-arithmetic experiments.
Token vectors (vision 257x1024, text 77x768) live in each encoder's own space
and are NOT directly comparable across modalities without projection.
"""
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from core.config import load_config


class ClipWrapper:
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        c = cfg["clip"]
        self.device = c["device"] if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if c["dtype"] == "float16" and self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(c["model_dir"], torch_dtype=dtype).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(c["model_dir"])
        self.save_tokens = c.get("save_token_vectors", True)

    @torch.no_grad()
    def encode_images(self, pil_images):
        """-> dict(embeds [N,768] L2-normalized float32, tokens [N,257,1024] or None)"""
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"].to(self.model.dtype))
        pooled = self.model.visual_projection(vision_out.pooler_output)
        embeds = torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy()
        tokens = vision_out.last_hidden_state.float().cpu().numpy() if self.save_tokens else None
        return {"embeds": embeds, "tokens": tokens}

    @torch.no_grad()
    def encode_texts(self, texts):
        """-> dict(embeds [N,768] L2-normalized float32, tokens [N,L,768] or None)"""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        text_out = self.model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        pooled = self.model.text_projection(text_out.pooler_output)
        embeds = torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy()
        tokens = text_out.last_hidden_state.float().cpu().numpy() if self.save_tokens else None
        return {"embeds": embeds, "tokens": tokens}

    @staticmethod
    def cosine(a, b):
        """Cosine similarity matrix between row vectors of a [N,d] and b [M,d]."""
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return a @ b.T
