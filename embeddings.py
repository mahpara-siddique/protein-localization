"""
Extract protein language model embeddings.
- ESM-2 (650M params → 1280D per protein)
- ProtT5-XL (half-precision → 1024D per protein)
Both use mean pooling over residue embeddings.
"""
import torch
import numpy as np
import re
from tqdm import tqdm
from config import (
    DEVICE, ESM_MODEL, ESM_LAYER, ESM_DIM, ESM_BATCH,
    PROTT5_MODEL, PROTT5_TOKENIZER, PROTT5_DIM, PROTT5_BATCH, PROTT5_MAX_LEN
)


def extract_esm2(sequences, save_path=None):
    """
    Extract ESM-2 embeddings via mean pooling.
    Returns: numpy array of shape (N, 1280)
    """
    print(f"\n  Loading ESM-2 model: {ESM_MODEL}")
    print(f"  Device: {DEVICE}")
    print(f"  Sequences: {len(sequences)}")

    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(DEVICE).eval()

    all_embeddings = []
    # ESM-2 max sequence length is ~1022 tokens
    data = [(f"seq_{i}", seq[:1022]) for i, seq in enumerate(sequences)]

    for i in tqdm(range(0, len(data), ESM_BATCH), desc="  ESM-2"):
        batch = data[i:i + ESM_BATCH]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(DEVICE)

        with torch.no_grad():
            results = model(tokens, repr_layers=[ESM_LAYER])
            token_embs = results["representations"][ESM_LAYER]  # [B, L, 1280]

            # Mean pool over actual residues (skip BOS token at position 0)
            for j, (_, seq) in enumerate(batch):
                seq_len = min(len(seq), token_embs.shape[1] - 2)
                emb = token_embs[j, 1:1 + seq_len, :].mean(dim=0)  # [1280]
                all_embeddings.append(emb.cpu().numpy())

        # Free GPU memory periodically
        if (i // ESM_BATCH) % 20 == 0:
            torch.cuda.empty_cache()

    result = np.array(all_embeddings, dtype=np.float32)

    if save_path:
        np.save(save_path, result)
        print(f"  Saved: {result.shape} → {save_path}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return result


def extract_prott5(sequences, save_path=None):
    """
    Extract ProtT5-XL embeddings via masked mean pooling.
    Returns: numpy array of shape (N, 1024)
    """
    from transformers import T5EncoderModel, T5Tokenizer

    print(f"\n  Loading ProtT5 model: {PROTT5_MODEL}")
    print(f"  This takes a few minutes to download (~3GB) on first run...")
    print(f"  Device: {DEVICE}")
    print(f"  Sequences: {len(sequences)}")

    tokenizer = T5Tokenizer.from_pretrained(PROTT5_TOKENIZER, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL)
    model = model.to(DEVICE).eval()

    # ProtT5 requires: space-separated amino acids, rare AAs replaced with X
    processed = []
    for seq in sequences:
        seq = seq[:PROTT5_MAX_LEN]
        seq = re.sub(r"[UZOB]", "X", seq)
        processed.append(" ".join(list(seq)))

    all_embeddings = []

    for i in tqdm(range(0, len(processed), PROTT5_BATCH), desc="  ProtT5"):
        batch = processed[i:i + PROTT5_BATCH]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=PROTT5_MAX_LEN
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            embs = outputs.last_hidden_state  # [B, L, 1024]

            # Masked mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
            masked_embs = embs * mask
            pooled = masked_embs.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, 1024]
            all_embeddings.append(pooled.cpu().numpy())

        if (i // PROTT5_BATCH) % 10 == 0:
            torch.cuda.empty_cache()

    result = np.vstack(all_embeddings).astype(np.float32)

    if save_path:
        np.save(save_path, result)
        print(f"  Saved: {result.shape} → {save_path}")

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    return result