

"""
Generating word Embeddings:
    Techiniques : 
        1. Word2Vec ( Mikolov Google 2013)
        2. GloVe ( Global Vectors for word representation) Stanford 2014 
    SemEval 2017 Competition:
        tasks : Semantic comparision (Given two sentences return score or 0 - 5
                                          based on the same meaning)
                Sentiment , humour and truth
                Parsing semantic structures
                
    Cross entropy: - sum( p(y actuals ) * p(y predicted))
    perplexity : 2 ** cross entropy
    A perfect model has cross entropy of 0 and perplexity of 1
"""

"""
Implementing RNN for character prediction for generating text

"""
from bs4 import BeautifulSoup

import requests

import random


"""
Downloading the data 
    1. All articles in the arxiv website related to required categories
    2. Movie conversation from cornel movie data set
"""
base_path = 'http://export.arxiv.org/api/query'

Categories = [
        'Machine Learning',
        'Neural and Evolutionary Computing',
        'Optimization'
        ]

Keywords = [
        'neural',
        'network',
        'deep']

def build_url(amount, offset):
    categories = ' OR '.join('cat:' + x for x in Categories)
    keywords = ' OR '.join('all:' + x for x in Keywords)
    
    url = base_path
    url += '?search_query=(({}) AND ({}))'.format(categories, keywords)
    url += '&max_results={}&offset={}'.format(amount, offset)
    
    return url
    
build_url(0,0)

def get_count():
    url = build_url(0,0)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    count = int(soup.find('opensearch:totalresults').string)
    print(count, 'papers found')
    return count
num_papers = get_count()
num_papers

page_size = 100

def fetch_page(amount, offset):
    url = build_url(amount,offset)
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    
    for entry in soup.findAll('entry'):
        text = entry.find('summary').text
        text = text.strip().replace('\n',' ')
        
        yield text


def fetch_all():
    for offset in range(0, num_papers, page_size):
        print('Fetch papers {}/{}'.format(offset+ page_size, num_papers ))
        
        for page in fetch_page(page_size, offset):
            yield page

download_filename = 'arxiv_abstracts.txt'

def download_data():
    if not os.path.exists(download_filename):
        with open(download_filename, 'w') as file:
            for abstract in fetch_all():
                file.write(abstract +'\n')
                
    with open(download_filename) as file:
        data = file.readlines()
        
    return data

data = download_data()

len(data)

"""
Optional: 
    Adding cornel movie conversational data

"""


with open('movie_lines.txt') as file:
    movie_lines_raw = file.readlines()

len(movie_lines_raw)

lines = [line.split(" +++$+++ ")[-1].replace('\n',"").lower() for line in movie_lines_raw]

"""
Making a paragaph for every 10 conversations:
"""
def converting_lines_paragraph(lines, num_lines_in_paragraph):
    paragraphs = []
    
    for i in range(int(len(lines)/ int(num_lines_in_paragraph)) -1):
        temp = ''
        for j in range(int(i*20) , int( (i+1)*20 )):
            temp += lines[j]
        paragraphs.append(temp)
    return paragraphs

paragraphs = converting_lines_paragraph(lines, 20)
"""
adding paragraphs to the data
"""

data = data + paragraphs
data[0]

############################################################################
"""
Data preprocessing:
"""

max_sequence_len = 50
batch_size = 100

vocabulary = " $%^&*(){}'""/?.>,<-_+|1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
len(vocabulary)
lookup ={x: i for i, x in enumerate(vocabulary)}

def one_hot(batch, sequence_len = max_sequence_len):
    one_hot_batch = np.zeros( (len(batch), sequence_len,len(vocabulary))) 
    
    for index, line in enumerate(batch):
        line = [x for x in line if x in lookup]
        assert 2 <= len(line) <= max_sequence_len
        
        for offset, character in enumerate(line):
            code = lookup[character]
            one_hot_batch[index, offset, code] = 1
            
    return one_hot_batch

def next_batch():
    
    windows = []
    
    for line in data:
        for i in range(0, len(line) - max_sequence_len +1, max_sequence_len //2):
            windows.append(line[i: i + max_sequence_len])
            
    assert all( len(x) == len(windows[0]) for x in windows )
    
    # print("Number of windows", len(windows))
    # print("length of one windows", len(windows[0]))
    
    
    while True:
        random.shuffle(windows)
        for i in range(0 , len(windows), batch_size):
            batch = windows[i: i + batch_size]
            
            yield one_hot(batch)
            

test_batch = None
for batch in next_batch():
    test_batch = batch
    print(batch.shape)
    break



tf.reset_default_graph()

sequence = tf.placeholder( tf.float32, [None, max_sequence_len, len(vocabulary)])

x = tf.slice(sequence, (0,0,0), (-1, max_sequence_len -1, -1))

y = tf.slice( sequence, (0,1,0), (-1,-1,-1))

x.shape

y.shape

def get_mask(target):
    mask = tf.reduce_max(tf.abs(target), reduction_indices = 2)
    return mask

def get_sequence_len(target):
    mask = get_mask(target)
    sequence_len = tf.reduce_sum(mask, reduction_indices = 1)
    
    return sequence_len

num_neurons = 300
cell_layers = 2

num_steps = max_sequence_len  - 1
num_classes = len(vocabulary)

sequence_len = get_sequence_len(y)

sequence_len


###############################################################################

def build_rnn(data, num_steps, sequence_len, initial = None):
    
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.GRUCell(num_neurons) for _ in range(cell_layers)])
    
    output, state = tf.nn.dynamic_rnn(
            inputs = data, 
            cell = multi_cell,
            dtype = tf.float32,
            initial_state = initial,
            sequence_length = sequence_len)
    
    weight = tf.Variable( tf.truncated_normal([num_neurons, num_classes],
                                              stddev = 0.01))
    bias = tf.Variable(tf.constant(0.1, shape = [num_classes]))
     
    flattened_output = tf.reshape(output, [-1, num_neurons])
    prediction = tf.nn.softmax(tf.matmul(flattened_output, weight) + bias)
    
    prediction= tf.reshape( prediction, [-1, num_steps, num_classes])
    
    return prediction , state

prediction, _ = build_rnn( x, num_steps, sequence_len)

mask =  get_mask(y)

prediction = tf.clip_by_value( prediction, 1e-10, 1.0)

cross_entropy = y * tf.log(prediction)
cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices = 2)
cross_entropy += mask

length = tf.reduce_sum( sequence_len,0)

cross_entropy = tf.reduce_sum( cross_entropy, reduction_indices=1)/length
cross_entropy = tf.reduce_mean(cross_entropy)


logprob = tf.multiply( prediction, y)

logprob = tf.reduce_max(logprob, reduction_indices= 2)

logprob = tf.log(tf.clip_by_value(logprob, 1e-10,1.0 ))/tf.log(2.0)

logprob += mask 
length = tf.reduce_sum(sequence_len, 0)
logprob = tf.reduce_sum( logprob, reduction_indices= 1) /length

logprob = tf.reduce_mean(logprob)

optimizer = tf.train.RMSPropOptimizer(0.002)

gradient = optimizer.compute_gradients(cross_entropy)

optimize = optimizer.apply_gradients(gradient)

num_epochs = 10
epoch_size = 50

logprob_evals = []

checkpoint_dir =    'new_sample_checkpoint_output'

"""
Training the model:
"""

with tf.Session() as ses:
    
    saver = tf.train.Saver()
    ses.run(tf.global_variables_initializer())
    
    for epoch in range( num_epochs):
        for _ in range(epoch_size):
            
            batch = next(next_batch())
            
            logprob_eval, _ = ses.run( (logprob, optimize), {sequence: batch})
            
            logprob_evals.append( logprob_eval)
        saver.save(ses, os.path.join(checkpoint_dir, 'char_pred'), epoch)
        
        perplexity = 2 ** -(sum(logprob_evals[-epoch_size:])/  epoch_size )
        
        print('Epoch {:2d} perplexity {:5.4f}'.format(epoch, perplexity))
        
    print("training completed!") 



###############################################################################
"""
RNN for prediction
Checking the model
"""
seq_len = 2


num_neurons = 300

cell_layers = 2

def build_rnn(data, num_steps, sequence_len, initial = None):
    
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.GRUCell(num_neurons) for _ in range(cell_layers)])
    
    output, state = tf.nn.dynamic_rnn(
            inputs = data, 
            cell = multi_cell,
            dtype = tf.float32,
            initial_state = initial,
            sequence_length = sequence_len)
    
    weight = tf.Variable( tf.truncated_normal([num_neurons, num_classes],
                                              stddev = 0.01))
    bias = tf.Variable(tf.constant(0.1, shape = [num_classes]))
     
    flattened_output = tf.reshape(output, [-1, num_neurons])
    prediction = tf.nn.softmax(tf.matmul(flattened_output, weight) + bias)
    
    prediction= tf.reshape( prediction, [-1, num_steps, num_classes])
    
    return prediction , state

tf.reset_default_graph()

sequence = tf.placeholder( tf.float32, [1, seq_len, len(vocabulary)])



x = tf.slice(sequence, (0,0,0), (-1, seq_len -1, -1))

y = tf.slice( sequence, (0,1,0), (-1,-1,-1))

state1 = tf.placeholder(tf.float32, [1, num_neurons])

state2 = tf.placeholder(tf.float32,[1, num_neurons])

state1

state2

sequence_length = get_sequence_len(y)

prediction, output = build_rnn(x, num_steps = seq_len -1 , 
                               sequence_len = sequence_length, 
                               initial = (state1, state2))


ses = tf.Session()

checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

checkpoint.model_checkpoint_path


if checkpoint and checkpoint.model_checkpoint_path:
    tf.train.Saver().restore(ses, checkpoint.model_checkpoint_path)
    
gen_seed ="We"
gen_len = 200

curr_state1 = np.zeros( (1, num_neurons))
curr_state2 = np.zeros((1, num_neurons))

gen_text = gen_seed

sampling_temperature = 0.1


def sample(dist):
    dist = np.log(dist)/sampling_temperature
    dist = np.exp(dist) / np.exp(dist).sum()
    
    choice = np.random.choice(len(dist), p = dist)
    choice = vocabulary[choice]
    
    return choice

for _ in range(gen_len):
    feed = {
            state1: curr_state1,
            state2: curr_state2,
            sequence: one_hot([gen_text[-1] + '?' ], sequence_len = seq_len)
            }

    gen_prediction_eval, (curr_state1, curr_state2) = ses.run([prediction, output],
                         feed)
    
    gen_text  += sample(gen_prediction_eval[0, 0])
    
    
"""
Output from the model:
"""
gen_text

