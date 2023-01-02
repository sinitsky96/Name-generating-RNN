import torch
import torch.nn as nn
from train_model_q2 import RNN, generate, test
import os
import unicodedata
import string
import glob
import random
import torch
import matplotlib.pyplot as plt

# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


# Turn a Unicode string to plain ASCII in order to get rid of accents
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )


def find_files(path):
    return glob.glob(path)


# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicode_to_ascii(line.strip()) for line in some_file]


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][ALL_LETTERS.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [ALL_LETTERS.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(N_LETTERS - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


def evaluate_model_q2():
    # Load the model
    rnn = RNN(N_LETTERS, 256, N_LETTERS)
    criterion = nn.NLLLoss()
    current_test_loss = 0
    current_test_acc = 0
    rnn.load('model_q2.pkl')
    rnn.eval()
    with torch.no_grad():
        for i in range(1000):
            category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
            _, loss, acc = test(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion)

            current_test_loss += loss
            current_test_acc += acc

    print("Test loss: ", current_test_loss / 1000)
    print("Test accuracy: ", current_test_acc / 1000)

    # Generate some names
    for i in range(5):
        # choose a random letter and category
        letter = random.choice(string.ascii_letters)
        category = random.choice(all_categories)
        print(f"letter : {letter}\n category : {category} \n name : {generate(rnn, category, letter)}")


if __name__ == '__main__':
    evaluate_model_q2()
