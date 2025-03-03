#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import pickle
import nltk
from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt')

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None

    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))

    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1

    def stop(self):
        sys.stdout.write("]\n")


startToken = '<S>'
startTokenIdx = 0

endToken = '</S>'
endTokenIdx = 1

unkToken = '<UNK>'
unkTokenIdx = 2

padToken = '<PAD>'
padTokenIdx = 3

transToken = '<TRANS>'
transTokenIdx = 4


def read_corpus(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print('Loading file:',fileName)
    with open(fileName, 'r', encoding="utf-8") as file:
        res = [list(line.rstrip().replace('\u200b', '')) for line in file]

        return res


def save_data(data, fileName):
    with open(fileName, "wb") as file:
        pickle.dump(data, file)


def load_data(fileName):
    with open(fileName, "rb") as file:
        return pickle.load(file)


def read_corpus_for_bleu(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print('Loading file:',fileName)
    return [nltk.word_tokenize(line) for line in open(fileName, encoding="utf-8")]