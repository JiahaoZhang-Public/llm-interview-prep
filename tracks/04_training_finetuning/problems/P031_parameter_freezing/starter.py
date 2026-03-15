def freeze_parameters(model, trainable_patterns):
    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = []
    for name, param in model.named_parameters():
        for pattern in trainable_patterns:
            if pattern in name:
                param.requires_grad = True
                trainable_names.append(name)
                break

    return trainable_names
