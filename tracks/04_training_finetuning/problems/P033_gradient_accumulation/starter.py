import torch
import torch.nn.functional as F


def run_gradient_accumulation(model, batches, optimizer, accumulation_steps: int):
    optimizer.zero_grad()
    step_count = 0

    for i, batch in enumerate(batches):
        logits = model(batch["inputs"])
        loss = F.cross_entropy(logits, batch["labels"])
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

    return step_count
