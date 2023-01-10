import torch

def cosine_score(batch, proto):
    ref = []
    for label in batch['label']:
        ref.append(torch.tensor(proto[label]))
    torch.stack(ref)
    