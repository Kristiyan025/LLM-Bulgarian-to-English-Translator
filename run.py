#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import os
import random
import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import model
import utils
from tokenizer import BPETokenizer
from training import train_model, perplexity

from parameters import *

if len(sys.argv) > 1 and sys.argv[1] == 'train_tokenizer':
    if os.path.exists(ind2tokenFileName) and os.path.exists(token2indFileName) and os.path.exists(mergesFileName):
        print("Tokenizer is already trained. Delete the tokenizer files to retrain.")
        sys.exit(0)

    wholeCorpus = (
            utils.read_corpus(sourceFileName)
            + utils.read_corpus(sourceDevFileName)
            + utils.read_corpus(targetFileName)
            + utils.read_corpus(targetDevFileName)
    )

    tokenizer = BPETokenizer()
    tokenizer.train(wholeCorpus, log=True)

    utils.save_data(tokenizer.ind2token, ind2tokenFileName)
    utils.save_data(tokenizer.token2ind, token2indFileName)
    utils.save_data(tokenizer.merges, mergesFileName)

elif len(sys.argv) > 1 and sys.argv[1] == 'tokenize_corpus':
    if os.path.exists(trainFileName) and os.path.exists(validationFileName) and os.path.exists(testFileName):
        print("Train, validation and test tokenized corpora already exist. Delete the files to recalculate.")
        sys.exit(0)

    tokenizer = BPETokenizer.from_files()
    train_corpus = tokenizer.tokenize_corpus(sourceFileName, targetFileName)
    print('Train corpus tokenizing completed.')
    validation_corpus = tokenizer.tokenize_corpus(sourceDevFileName, targetDevFileName)
    print('Validation corpus tokenizing completed.')
    test_corpus = tokenizer.tokenize_corpus(sourceTestFileName, targetTestFileName)
    print('Test corpus tokenizing completed.')

    utils.save_data(train_corpus, trainFileName)
    utils.save_data(validation_corpus, validationFileName)
    utils.save_data(test_corpus, testFileName)

elif len(sys.argv) > 1 and sys.argv[1] in ('train', 'extratrain'):
    if not os.path.exists(trainFileName) or not os.path.exists(validationFileName):
        print("Train and/or validation tokenized corpora are missing. Tokenize the corpora first.")
        sys.exit(0)

    trainCorpus = utils.load_data(trainFileName)
    valCorpus = utils.load_data(validationFileName)
    # trainCorpus = random.sample(trainCorpus, len(trainCorpus) // 4)

    train_model(
        modelFileName,
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
        extra_train=sys.argv[1] == 'extratrain',
    )

elif len(sys.argv) > 3 and sys.argv[1] == 'perplexity':
    tokenizer = BPETokenizer.from_files()

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
    nmt.load(modelFileName)

    sourceTest = utils.read_corpus(sys.argv[2])
    targetTest = utils.read_corpus(sys.argv[3])
    test = ((tokenizer.tokenize(s), tokenizer.tokenize(t)) for s, t in zip(sourceTest,targetTest))
    test = [[utils.startTokenIdx] + s + [utils.transTokenIdx] + t + [utils.endTokenIdx] for s, t in test]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, test, batch_size))

elif len(sys.argv) > 3 and sys.argv[1] == 'translate':
    tokenizer = BPETokenizer.from_files()

    sourceTest = utils.read_corpus(sys.argv[2])
    test = (tokenizer.tokenize(s) for s in sourceTest)
    test = [[utils.startTokenIdx] + s + [utils.transTokenIdx] for s in test]

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
    nmt.load(modelFileName)

    nmt.eval()
    file = open(sys.argv[3], 'w', encoding="utf-8")
    pb = utils.progressBar()
    pb.start(len(test))
    for s in test:
        r = list(nmt.generate(s, temperature=temperature))
        st = r.index(utils.transTokenIdx)
        result = tokenizer.untokenize(r[st + 1:-1])
        file.write(''.join(result) + "\n")
        pb.tick()
    pb.stop()

elif len(sys.argv) > 2 and sys.argv[1] == 'generate':
    tokenizer = BPETokenizer.from_files()

    test = tokenizer.guarded_tokenize(list(sys.argv[2]))

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
    nmt.load(modelFileName)

    print("Model loaded.")
    nmt.eval()
    # result = nmt.generate(test)
    # result = tokenizer.untokenize(result)
    # print(''.join(result)+"\n")
    for token in nmt.generate(test, temperature=temperature):
        print(tokenizer.untokenize([token])[0], end='')
    print()

elif len(sys.argv) > 3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.read_corpus_for_bleu(sys.argv[2])]
    hyp = utils.read_corpus_for_bleu(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
