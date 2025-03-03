import math
import time

import numpy as np
import torch

import model
from parameters import *


def update_learning_rate(optimizer, iter, warmup_steps, d_model):
    lr = (d_model ** (-0.5)) * min(iter ** (-0.5), iter * warmup_steps ** (-1.5))
    if iter < warmup_steps:
        lr = max(0.0003, lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def perplexity(nmt, test, batchSize):
    testSize = len(test)
    H = 0.
    c = 0
    for b in range(0,testSize,batchSize):
        batch = test[b:min(b+batchSize, testSize)]
        l = sum(len(s) - 1 for s in batch)
        c += l
        with torch.no_grad():
            padded_batch = nmt.preparePaddedBatch(batch)
            _, loss = nmt(padded_batch[:, :-1], targets=padded_batch[:, 1:])
            H += l * loss
    return math.exp(H/c)


def train_model(
        model_filename,
        trainCorpus,
        valCorpus,
        num_blocks,
        d_model,
		num_heads,
		d_k,
		d_v,
		dropout,
		d_ff,
        label_smoothing,
        learning_rate,
        batch_size,
        clip_grad,
        max_epochs,
        warmup_steps,
        log_every,
        test_every,
        extra_train: bool = False,
):
    nmt = model.LanguageModel(
        max_seq_len,
        num_blocks,
        d_model,
        num_heads,
        d_k,
        d_v,
        dropout,
        d_ff,
        label_smoothing,
    ).to(device)

    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    if extra_train:
        nmt.load(model_filename)
        (iter,bestPerplexity,learning_rate,osd) = torch.load(model_filename + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpus), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(max_epochs):
        np.random.shuffle(idx)
        tokens = 0
        trainTime = time.time()
        for b in range(0, len(idx), batch_size):
            iter += 1
            update_learning_rate(optimizer, iter, warmup_steps=warmup_steps, d_model=d_model)

            batch = [trainCorpus[i] for i in idx[b:min(b + batch_size, len(idx))]]

            tokens += sum(len(s) - 1 for s in batch)
            padded_batch = nmt.preparePaddedBatch(batch)
            _, loss = nmt(padded_batch[:, :-1], targets=padded_batch[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            if iter % log_every == 0:
                print(
                    "Iteration:", iter,
                    "Epoch:", epoch + 1, '/', max_epochs,
                    "Batch:", b // batch_size + 1, '/', len(idx) // batch_size + 1,
                    "Loss:", loss.item(),
                    "tokens/sec:", tokens / (time.time() - trainTime),
                    "Time elapsed:", (time.time() - beginTime), "s"
                )

                trainTime = time.time()
                tokens = 0

            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, valCorpus, batch_size)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                nmt.save(model_filename)
                torch.save((iter, bestPerplexity, learning_rate, optimizer.state_dict()),
                           model_filename + '.optim')

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(model_filename + "_best")
                    torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), model_filename + '_best.optim')


    print('Reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, valCorpus, batch_size)
    print('Last model perplexity: ',currentPerplexity)

    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(model_filename)
        torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), model_filename + '.optim')