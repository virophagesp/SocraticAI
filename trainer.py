import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import json
import gc
import ctypes

# These constants are the hyperparameters
# The batch size is how many independent sequences will we process in parallel
BATCH_SIZE = 64
# The maximum context length for predictions
BLOCK_SIZE = 256
# Maximum interations of training
MAX_ITERS = 2000
# Beginning model's amount of iterations
BEGIN_INTERATIONS = 0
# Every 100 iterations of training, print loss and save stage
EVAL_INTERVAL = 100
# The rate of learning
LEARNING_RATE = 3e-4
# I test this on my personal laptop which does not have a gpu so the cpu will be used
DEVICE = 'cpu'
EVAL_ITERS = 200
# The dimension of embedding
N_EMBD = 384
# The number of self-attention heads
N_HEAD = 6
# The number of block layers
N_LAYER = 6

# The size of self-attention heads
HEAD_SIZE = N_EMBD // N_HEAD


# Open and read the raw training and testing data from the JSON files
raw_training_data = json.load(open('questions.json', 'r'))
raw_testing_data = json.load(open('testing.json', 'r'))

# file that records testing
testing_file = open('testing.txt', 'w')


def print_and_write_to_file(string_input):
    """ print and write to testing file """

    print(string_input)
    testing_file.write(f'{string_input}\n')


# Manually set seed for ai for consistent results during testing and such
torch.manual_seed(1337)

# Here are all the words that are common for every question and answer
# they will be converted to singular tokens
# to improve performance without increasing block size
words = [
    'question: "',
    'what',
    'why',
    'where',
    'how',
    'should',
    '?"\nanswer: "i don\'t know',
    'could',
    'you',
    'explain',
    'the',
    'concept',
    'of'
]
# within all the data, each known letter, number, and special character, as a token
chars = ''
for data in raw_training_data:
    if type(data) == list:
        chars += ''.join(data)
    else:
        chars += data
for data in raw_testing_data:
    if type(data) == list:
        chars += ''.join(data)
    else:
        chars += data
chars += ''.join(words)
# remove duplicate characters and converts the string into a sorted list
chars = sorted(list(set(chars)))
# all tokens
vocabulary = chars + words

# size of vocabuary
vocab_size = len(vocabulary)
# create encoding mapping from characters to integers
# and decoding mapping from integers to characters
vocab_to_int = {}
int_to_vocab = {}
for i, ch in enumerate(vocabulary):
    vocab_to_int[ch] = i
    int_to_vocab[i] = ch


def filter(string):
    """ filter: take a string as input, output that same string
    but without any characters that aren't in the vocabuary"""

    output = ""
    string_index = 0
    while string_index < len(string):
        # if character not part of vocabuary, skip it
        if string[string_index] in chars:
            output += string[string_index]
        string_index += 1
    # only 1 question mark at end
    while output[-1] == '?':
        output = output[:-1]
    return output


def encode(string):
    """ encoder: take a string, output a list of integer tokens """

    tokens = []

    string_index = 0
    while string_index < len(string):
        word_index = 0

        # check if next part is a word in the list of words
        while word_index < len(words):
            word = words[word_index]
            if string[string_index: string_index + len(word)] == word:
                tokens.append(vocab_to_int[word])
                string_index += len(word)
                break
            word_index += 1

        # append single character token if no word was added
        if word_index == len(words):
            tokens.append(vocab_to_int[string[string_index]])
            string_index += 1

    return tokens


def decode(tokens):
    """ decoder: take a encoded list of integers representing tokens output the decoded string """

    decoded = ''.join([int_to_vocab[token] for token in tokens])
    return decoded


# process the raw json data and convert it to usable text
def raw_to_processed(raw_data):
    output_data = ""
    # Loop over data and write to input file
    data_looper = 0
    # only half the length since it will be using two items at a time
    while data_looper < int(len(raw_data)/2):
        # Get the question, every even number are the strings
        question = raw_data[data_looper*2]
        # Loop over the answers, every odd number are the lists
        for answer in raw_data[data_looper*2+1]:
            # Add the same amount of newline characters
            #  as block_size in the training script
            # The reason why this is done is because
            #  the ai will incorrectly learn to try
            #  to predict more questions after answering
            #  this not only is not something it should learn
            #  but could cause issues and confusion for users
            #  if it seemingly starts getting new weird inputs
            #  that weren't typed in
            for i in range(BLOCK_SIZE):
                output_data += '\n'
            # Print question then answer
            output_data += f'question: "{question}"\n'
            output_data += f'answer: "{answer}"\n'
        data_looper += 1
    return output_data


# Open the files containing output of using preprocessor
train_data = encode(raw_to_processed(raw_training_data).lower())
validation_data = encode(raw_to_processed(raw_testing_data).lower())


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self):
        """  """

        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * HEAD_SIZE, bias=False)
        self.tril = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, head_input):
        """  """

        # split the merged qkv layer into separate query key and value
        query, key, value = self.qkv(head_input).chunk(3, dim=-1) # all three are (1,BLOCK_SIZE,HEAD_SIZE)
        # compute attention scores ("affinities")
        weight = query @ key.transpose(-2, -1) * HEAD_SIZE**-0.5  # (1, BLOCK_SIZE, HEAD_SIZE) @ (B, HEAD_SIZE, BLOCK_SIZE) -> (1, BLOCK_SIZE, BLOCK_SIZE)
        weight = weight.masked_fill(
            self.tril[:BLOCK_SIZE, :BLOCK_SIZE] == 0,
            float('-inf')
        )  # (1, BLOCK_SIZE, BLOCK_SIZE)
        weight = F.softmax(weight, dim=-1)  # (1, BLOCK_SIZE, BLOCK_SIZE)
        # perform the weighted aggregation of the values
        output = weight @ value  # (1, BLOCK_SIZE, BLOCK_SIZE) @ (1, BLOCK_SIZE, HEAD_SIZE) -> (1, BLOCK_SIZE, HEAD_SIZE)
        return output


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        """  """

        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])
        self.project = nn.Linear(N_EMBD, N_EMBD)
        self.LayerNormal = nn.LayerNorm(N_EMBD)

    def forward(self, logits):
        """  """

        output = self.LayerNormal(logits)
        output = torch.cat([h(output) for h in self.heads], dim=-1)
        output = self.project(output)
        return output


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        """  """

        super().__init__()
        self.SelfAttention = MultiHeadAttention()
        self.FeedFoward = nn.Sequential(
            nn.LayerNorm(N_EMBD),
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, logits):
        """  """

        logits = logits + self.SelfAttention(logits)
        logits = logits + self.FeedFoward(logits)
        return logits


class GPTLanguageModel(nn.Module):

    def __init__(self):
        """  """

        super().__init__()
        # each token directly reads off the logits
        #  for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.FinalLayerNormal = nn.LayerNorm(N_EMBD)  # final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        # better init, not covered in the original GPT video
        #  note watch newer video that explains to make a better comment
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """  """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context):
        """  """

        # context and targets are both (1,BLOCK_SIZE) tensor of integers
        token_embed = self.token_embedding_table(context)  # (1,BLOCK_SIZE,N_EMBD)
        position_embed = self.position_embedding_table(
            torch.arange(BLOCK_SIZE, device=DEVICE)
        )  # (BLOCK_SIZE,N_EMBD)
        logits = token_embed + position_embed  # (1,BLOCK_SIZE,N_EMBD)
        logits = self.blocks(logits)  # (1,BLOCK_SIZE,N_EMBD)
        logits = self.FinalLayerNormal(logits)  # (1,BLOCK_SIZE,N_EMBD)
        logits = self.lm_head(logits)  # (1,BLOCK_SIZE,vocab_size)

        return logits


def batch(model, data):
    """ get a batch of training data """

    # load block sized chunks of a batch of the data for inputs context and targets
    context = torch.zeros([BATCH_SIZE, BLOCK_SIZE], dtype=torch.long, device=DEVICE)
    targets = torch.zeros([BATCH_SIZE, BLOCK_SIZE], dtype=torch.long, device=DEVICE)
    for batch_index in range(BATCH_SIZE):
        data_batch = torch.randint(len(data) - BLOCK_SIZE, (1,))[0]
        for block_index in range(BLOCK_SIZE):
            context[batch_index][block_index] = data[data_batch + block_index]
            targets[batch_index][block_index] = data[data_batch + block_index + 1]

    logits = model.forward(context)

    logits = logits.view(BATCH_SIZE * BLOCK_SIZE, vocab_size)
    targets = targets.view(BATCH_SIZE * BLOCK_SIZE)
    loss = F.cross_entropy(logits, targets)

    return loss


def generate(model, context):
    """ generate tokens after  """

    # context is initally (1, BLOCK_SIZE) array of indices in the current context
    while context[0].tolist()[-1] != vocab_to_int['"']:
        # crop context to the last BLOCK_SIZE tokens
        context_crop = context[:, -BLOCK_SIZE:]
        # get the predictions
        logits = model.forward(context_crop)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (1, vocab_size)
        # apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=-1)  # (1, vocab_size)
        # sample from the distribution
        context_next_part = torch.multinomial(probabilities, num_samples=1)  # (1, 1)
        # append sampled index to the running sequence
        context = torch.cat((context, context_next_part), dim=1)  # (1, current context length + 1)

    return context


# this property improves performance for this function
@torch.no_grad()
def estimate_loss(model, step, model_index):
    """ estimate the model's loss """

    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for looper in range(EVAL_ITERS):
        losses[looper] = batch(model, train_data).item()
    out_train = losses.mean()
    losses = torch.zeros(EVAL_ITERS)
    for looper in range(EVAL_ITERS):
        losses[looper] = batch(model, validation_data).item()
    out_validate = losses.mean()
    model.train()

    # print the step and losses
    print_and_write_to_file(f'step {step}: train loss {out_train:.4f}')
    # this is to align it with train loss during printing
    aligment_spacing = ' ' * len(str(step))
    print_and_write_to_file(f'  {aligment_spacing}  validate loss {out_validate:.4f}')

    # save model
    with open(f'model{model_index}.pkl', 'wb') as f:
        pickle.dump(model, f)


def model_information_printer(parameters, mode_index=None):
    """ print the model's parameters and index if in testing mode """

    # get number of parameters
    parameter_count = 0
    for parameter in parameters:
        parameter_count += parameter.numel()

    # print the number of parameters measured in millions in the model
    # and it's index
    print_and_write_to_file('')
    if mode_index is None:
        print_and_write_to_file(f'model has {parameter_count/1e6} million parameters')
    else:
        print_and_write_to_file(f'model {mode_index} has {parameter_count/1e6} million parameters')
    print_and_write_to_file('')


def question_answerer(model, question_string):
    """ input question and generate answer """

    # encode question
    question = encode(f'question: "{filter(question_string)}?"\nanswer: "i don\'t know')
    # when the block is not at least BLOCK_SIZE, it will crash
    # add newline characters as placeholders, like the training data
    while len(question) < BLOCK_SIZE:
        question.insert(0, vocab_to_int['\n'])

    # generate from the model
    context = torch.zeros(
        (1, len(question)),
        dtype=torch.long,
        device=DEVICE
    )
    for looper in range(len(question)):
        context[0][looper] = question[looper]
    generated = generate(model, context)

    # blank line between separate sets of questions and answers
    print_and_write_to_file('')

    # generated only has 1 element, this is the string with the question and answer
    output_as_string = decode(generated[0].tolist())

    # make sure there are not multiple newlines between separate sets of questions and answers
    while output_as_string[0] == '\n':
        output_as_string = output_as_string[1:]
    while '\n\n' in output_as_string:
        output_as_string = output_as_string.replace('\n\n', '')

    # print question and answer
    print_and_write_to_file(output_as_string)


# free torch memory usage on cpu mode after training done
def trim_memory():
  libc = ctypes.CDLL("libc.so.6")
  return libc.malloc_trim(0)


# Ask user if they want to retrain the models
mode = input('retrain the models (y/n): ').lower()
testing_file.write(f'retrain the models (y/n): {mode}\n')

# train the model
if mode == 'y':
    # index for stages
    model_index = int(BEGIN_INTERATIONS/EVAL_INTERVAL)

    # create the 0th model if doesn't exist
    if BEGIN_INTERATIONS == 0:
        model = GPTLanguageModel()
        model.to(DEVICE)
        with open(f'model0.pkl', 'wb') as f:
            pickle.dump(model, f)
    # load model
    else:
        with open(f'model{int(BEGIN_INTERATIONS/EVAL_INTERVAL)}.pkl', 'rb') as f:
            model = pickle.load(f)

    # print the number of parameters in the model
    model_information_printer(model.parameters())

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iteration in range(BEGIN_INTERATIONS, MAX_ITERS):

        # every EVAL_INTERVAL evaluate the loss on train and val sets
        if iteration % EVAL_INTERVAL == 0:
            estimate_loss(model, iteration, model_index)
            model_index += 1

        # evaluate the loss
        model_loss = batch(model, train_data)
        optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        optimizer.step()

    # estimate the loss of the final model
    estimate_loss(model, MAX_ITERS, int(MAX_ITERS/EVAL_INTERVAL))

    del model_loss
    del model
    del optimizer
    gc.collect()

    trim_memory()

# ask user if they want to test the models
mode = input('test the models (y/n): ').lower()
testing_file.write(f'test the models (y/n): {mode}\n')

# input question from user for only last model
if mode != 'y':
    # load model
    with open(f'model{int(MAX_ITERS/EVAL_INTERVAL)}.pkl', 'rb') as f:
        model = pickle.load(f)
    model.to(DEVICE)

    # print the number of parameters in the model
    model_information_printer(model.parameters())

    # input question from user and get answer from model
    user_question = input('question: ').lower()
    testing_file.write(f'question: {user_question}\n')
    question_answerer(model, user_question)

# test questions from file for the saved model stages
else:
    # load testing questions from file
    questions = json.load(open('testing.json', 'r'))

    # loop over each model file
    for model_index in range(int(BEGIN_INTERATIONS/EVAL_INTERVAL), int(MAX_ITERS/EVAL_INTERVAL)+1):
        # load model file
        with open(f'model{model_index}.pkl', 'rb') as f:
            model = pickle.load(f)
        model.to(DEVICE)

        # print the number of parameters in the model and it's index
        model_information_printer(model.parameters(), model_index)

        # test model with each question
        looper = 0
        while looper < int(len(questions)/2):
            question_answerer(model, questions[looper*2])
            looper += 1

        print_and_write_to_file('')
