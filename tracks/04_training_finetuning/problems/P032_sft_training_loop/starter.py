import torch
import torch.nn.functional as F


def sft_train_step(model, batch, optimizer):
    optimizer.zero_grad()
    logits = model(batch["inputs"])
    loss = F.cross_entropy(logits, batch["labels"])
    loss.backward()
    optimizer.step()
    return loss.item()
