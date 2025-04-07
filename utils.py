import torch
from torch import nn
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator
from datasets import load_dataset, Dataset
from transformers import PreTrainedModel, AutoTokenizer

from tqdm import tqdm
import faiss
import os

import requests
from PIL import Image
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
from typing import List, Tuple, Literal, Union

import random


def mine_hard_neg_multimodal(
    dataset: Dataset,
    anchor: str,
    positive: str,
    negative: str,
    num_negatives: int,
    embedding_model: Union[SentenceTransformer, PretrainedModel] = None,
    tokenizer: AutoTokenizer = None,
    min_range: int = 5,
    max_range: int = 50,
    mode: Literal["top", "random"] = "top",
    faiss_index=None,
    use_faiss: bool = True,
) -> Dataset:
    """
    Optimized function for mining hard negatives from a dataset, using FAISS for speed, with tqdm progress.
    Supports both SentenceTransformer and Hugging Face Transformers models.

    Args:
        dataset: The input dataset.
        embedding_model: The SentenceTransformer or Hugging Face Transformers model.
        tokenizer: Required if using a Hugging Face Transformers model.
        anchor: The anchor column name.
        negative: The negative column name.
        positive: The positive column name.
        num_negatives: The number of negatives to mine.
        min_range: The minimum range for negative selection.
        max_range: The maximum range for negative selection.
        mode: "top" or "random" selection mode.
        faiss_index: A pre-built FAISS index. If None and use_faiss is True, it will be built.
        use_faiss: Whether to use FAISS for similarity search.
    """

    df = dataset.to_pandas()
    all_neg_titles = []
    all_anchor_repeats = []
    all_positive_repeats = []
    all_title_embeddings = []

    def get_embedding(text):
        if isinstance(embedding_model, SentenceTransformer):
            return embedding_model.encode(text, convert_to_tensor=True).cpu().numpy()
        else:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128, padding=True
            )
            with torch.no_grad():
                outputs = embedding_model(**inputs.to(embedding_model.device))
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

    # Pre-encode all titles with tqdm progress
    for title in tqdm(df[positive], desc="Encoding Titles"):
        all_title_embeddings.append(get_embedding(title))
    all_title_embeddings = np.array(all_title_embeddings).astype("float32")

    if use_faiss:
        if faiss_index is None:
            embedding_dim = all_title_embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(
                embedding_dim
            )  # Inner product for cosine similarity
            faiss_index.add(all_title_embeddings)

    # Use tqdm to wrap the loop for progress visualization
    for index in tqdm(range(len(df)), desc="Mining Hard Negatives"):
        img_embed = (
            embedding_model.encode(
                Image.open(requests.get(df[anchor][index], stream=True).raw),
                convert_to_tensor=True,
            )
            if isinstance(embedding_model, SentenceTransformer)
            else get_embedding(
                Image.open(requests.get(df[anchor][index], stream=True).raw)
            )
        )

        img_embed = (
            torch.tensor(img_embed)
            if not isinstance(img_embed, torch.Tensor)
            else img_embed
        )

        if use_faiss:
            distances, indices = faiss_index.search(
                img_embed.cpu().numpy().reshape(1, -1),
                max(num_negatives + max_range, len(df)),
            )
            top_indices = torch.tensor(indices[0])
        else:
            title_embed = torch.tensor(all_title_embeddings)
            similarity = F.cosine_similarity(img_embed.unsqueeze(0), title_embed, dim=1)
            _, top_indices = torch.topk(
                similarity, k=max(num_negatives + max_range, len(df))
            )

        if mode == "top":
            if index in top_indices:
                neg_indexes = (
                    top_indices[top_indices != index][
                        min_range : min_range + num_negatives
                    ]
                    .cpu()
                    .numpy()
                    .tolist()
                )
            else:
                neg_indexes = (
                    top_indices[min_range : min_range + num_negatives]
                    .cpu()
                    .numpy()
                    .tolist()
                )

        elif mode == "random":
            population = (
                top_indices[top_indices != index][min_range:max_range]
                .cpu()
                .numpy()
                .tolist()
            )
            if len(population) > 0:
                neg_indexes = random.sample(
                    population, min(num_negatives, len(population))
                )
            else:
                neg_indexes = []

        neg_titles = [df[positive][i] for i in neg_indexes]
        all_neg_titles.extend(neg_titles)
        all_anchor_repeats.extend([df[anchor][index]] * len(neg_titles))
        all_positive_repeats.extend([df[positive][index]] * len(neg_titles))

    new_df = pd.DataFrame(
        {
            anchor: all_anchor_repeats,
            positive: all_positive_repeats,
            negative: all_neg_titles,
        }
    )

    return Dataset.from_pandas(new_df)
