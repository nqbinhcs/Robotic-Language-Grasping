"""
---
title: CLIP Text Embedder
summary: >
 CLIP embedder to get prompt embeddings for stable diffusion
---

# CLIP Text Embedder

This is used to get prompt embeddings for [stable diffusion](../index.html).
It uses HuggingFace Transformers CLIP model.
"""

from typing import List

import torch
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(
        self,
        version: str = "openai/clip-vit-base-patch16",
        device="cuda:0",
        max_length: int = 20,
    ):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version, device=device)

        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(version).to(device).eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """

        with torch.no_grad():
            # Tokenize the prompts
            batch_encoding = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            # Get token ids
            tokens = batch_encoding["input_ids"].to(self.device)
            feature = self.transformer(input_ids=tokens).last_hidden_state

        return feature[0]


## Example Usage
if __name__ == "__main__":
    prompts = ["grasp a cup"]
    embedder = CLIPTextEmbedder()
    embeddings = embedder(prompts)

    print(embeddings.shape)
    print(embeddings.device)
