import os
import unicodedata
import string
import glob
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

"""
##########Utility functions for preparing data into one-hot vectors and for training the network#######################
"""

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


"""
    ###################################Network definition#######################################################
"""


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output = self.dropout(output)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def predict(self, input_tensor, hidden_tensor):
        """
        predict the output
        :param input_tensor: input tensor
        :param hidden_tensor: hidden tensor
        :return: output tensor
        """
        output, hidden = self.forward(input_tensor, hidden_tensor)
        return output

    def load(self, path):
        """
        load the network
        :param path: path to load the network
        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
            save the network
            :param path: path to save the network
        """
        torch.save(self.state_dict(), path)


"""
    ###################################Prep for training#######################################################
"""


def letter_from_output(output):
    top_n, top_i = output.topk(1)
    top_i = top_i[0].item()
    letter = ALL_LETTERS[top_i]
    return letter


def train(model, category_tensor, input_line_tensor, target_line_tensor, criterion, learning_rate=0.005):
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item() / input_line_tensor.size(0)


"""
################################################prepare for testing####################################################
"""


def test(model, category_tensor, input_line_tensor, target_line_tensor, criterion):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden()
    model.zero_grad()
    loss = 0
    acc = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        if i<target_line_tensor.size(0)-1:
            acc += (output.argmax(dim=1) == target_line_tensor[i]).sum().item()
        loss += l

    return output, loss.item() / input_line_tensor.size(0), acc/(target_line_tensor.size(0)-1)


"""
    ###################################Training and Testing#######################################################
"""


def train_model_q2():
    rnn = RNN(N_LETTERS, 256, N_LETTERS)
    n_iters = 200000
    print_every = 5000
    plot_every = 500
    train_losses = []
    current_train_loss = 0  # Reset every plot_every iters
    test_losses = []
    current_test_loss = 0  # Reset every plot_every iters
    criterion = nn.NLLLoss()
    learning_rate = 0.0005

    for iter in range(1, n_iters + 1):
        category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
        _, loss = train(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion, learning_rate)
        current_train_loss += loss
        if iter % print_every == 0:
            print(' (%d %d%%) Loss : %.4f' % (iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            train_losses.append(current_train_loss / plot_every)
            current_train_loss = 0
            # test the model on test data for every 500 iterations
            rnn.eval()
            with torch.no_grad():
                for i in range(1000):
                    category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
                    _, loss,_ = test(rnn, category_tensor, input_line_tensor, target_line_tensor, criterion)
                    current_test_loss += loss
            test_losses.append(current_test_loss / 1000)
            current_test_loss = 0
            rnn.train()
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()
    rnn.save('model_q2.pkl')
    return rnn


def generate(model, category, start_letter='A'):
    """
    generate the name of the country
    :param model: model
    :param category:
    :param start_letter:
    :return:
    """
    max_length = 20
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = model.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = model(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            # check for EOS
            if topi == N_LETTERS - 1:
                break
            else:
                letter = ALL_LETTERS[topi]
                # add the letter to final string
                output_name += letter
            input = inputTensor(letter)

        return output_name


if __name__ == '__main__':
    rnn = train_model_q2()
