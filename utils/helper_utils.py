import torch
# Source: https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another

def prune_illegal_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
    return torch.utils.data.dataloader.default_collate(batch)

def batch_iter(dataloader, epochs):
    for epochId in range(epochs):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch