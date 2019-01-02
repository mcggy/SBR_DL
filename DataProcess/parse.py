'''
author:jiangyuan
data:2018/7/29
function: parse training data
'''

import re
import os, string
import time
import numpy as np
from collections import defaultdict
import json
import nltk
from Config import config
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from statistics import mean

# Read key words of java program language
fp = open(r'../resource/keyword.txt')
keyword = []
for line in fp.readlines():
    keyword.append(line.replace("\n", ""))
# fp.close()
# remove_punctuation_map = dict(ord(ord(char), ' ' + char + ' ') for char in string.punctuation)
# remove_number_map = dict((ord(char), " ") for char in string.digits)
#
# # Remove code snippet
def remove_code(text):
    # temp = text.translate(remove_punctuation_map)
    # temp = temp.translate(remove_number_map)
    list_words = word_tokenize(text)

    all = len(list_words)
    if (all == 0):
        return ''
    keywords = len([w for w in list_words if w in keyword])
    if (1.0 * keywords / all > 0.35):
        return ' '
    else:
        return text

# Processing the text of BugReport
def preprocess_br(raw_description):
    print(raw_description)
    if type(raw_description) == float:
        return ' '
    # 1. Remove \r
    current_desc = raw_description.replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                          current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 3.5 Remove Issue id : (only exists in chromium, others project can commented out)
    current_desc = re.sub(r'Issue (.*) : ', '', current_desc)
    # 4. Remove hex code
    current_desc = re.sub(r'(\w*)0x\w+', '', current_desc)
    # 5. Remove code snippet
    current_desc=remove_code(current_desc)
    # 6. Change to lower case
    current_desc = current_desc.lower()
    # 7. only letters
    letters_only = re.sub("[^a-zA-Z\.]", " ", current_desc)
    current_desc = re.sub("\.(?!((c|h|cpp|py)\s+$))", " ", letters_only)
    # 8. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    # 9. Remove stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    meaningful_words = [w for w in current_desc_tokens if not w in stopwords]
    # 10. Stemming
    snowball = SnowballStemmer("english")
    stems = [snowball.stem(w) for w in meaningful_words]
    return stems

# 处理summary或者description没有任何内容的情况
def dealNan(x):
    if type(x) == float:
        x = ' '
    return x

def clean_pandas(data):
    data['bug_title'] = data.bug_title.apply(dealNan)
    data['bug_description'] = data.bug_description.apply(dealNan)
    # 对文本数据进行清洗
    data['bug_title'] = data['bug_title'].map(lambda x: preprocess_br(x))
    data['bug_description'] = data['bug_description'].map(lambda x: preprocess_br(x))
    return data


# data processing and loading
def load_data(fname):
    # load df data
    file,_=os.path.splitext(fname)
    pandas_data_file=file+'.df'
    if os.path.exists(pandas_data_file):
        df=pd.read_pickle(pandas_data_file)
    else:
        df=pd.read_csv(fname)
        df=clean_pandas(df)
        df.to_pickle(pandas_data_file)
    ids = list(range(0, df.iloc[:, 0].size))
    summary=list(df['bug_title'])
    description=list(df['bug_description'])
    labels=list(df['bug_label'])

    return ids, summary, description,labels


# generate vocabulary
def gen_vocab(data):
    vocab = defaultdict(int)
    vocab_idx = 0
    for field in data:
        for text in field:
            for token in text:
                if token not in vocab:
                    vocab[token] = vocab_idx
                    vocab_idx += 1

    vocab['UNK'] = vocab_idx
    f = open(config.VOCAB_PATH, 'w')
    json.dump(vocab, f)
    return vocab


def get_maxlen(data):
    return max(map(lambda x: len(x), data))
def get_meanlen(data):
    return mean(map(lambda x: len(x), data))

# 将每个句子转化成定长max_len，并且句子中的词转化成vocab中对应的位置的id
def gen_seq(data, vocab, max_len):
    X = []
    for text in data:
        # 初始化一个list列表，长度是max_len
        temp = [0] * max_len
        # map(f,list) lambda表达式被当成函数表达式，text是list。vocab字典当返回的键值不存在时，返回默认的值vocab['UNK']
        temp[:len(text)] = map(lambda x: vocab.get(x, vocab['UNK']), text)

        if len(temp)>max_len:
            temp=temp[0:max_len]
        X.append(temp)

    X = np.array(X)
    return X

def train_word2vec(all_data):
    # Word2vec parameters
    min_word_frequency_word2vec = 5
    embed_size_word2vec = 200
    context_window_word2vec = 5
    # Learn the word2vec model and extract vocabulary
    wordvec_path = os.path.join(config.OUTPUT_PATH,
        '{}features_{}minwords_{}context.model'.format(embed_size_word2vec, min_word_frequency_word2vec,
                                                       context_window_word2vec))
    print('starts loading model.....')
    start = time.time()
    if os.path.exists(wordvec_path):
        # load word2vec model
        wordvec_model = Word2Vec.load(wordvec_path)
    else:
        wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec,
                                 window=context_window_word2vec)
        wordvec_model.save(wordvec_path)
    end = time.time()
    print('Finish load, Time-consuming%s s' % (end - start))


if __name__ == '__main__':
    print('parse data...')
    file_name = '../resource/chromium/chromium.csv'
    outdir = config.OUTPUT_PATH
    ids, summary,description, labels = load_data(file_name)

    # get vocabulary
    vocab = gen_vocab([summary,description])
    # get max length of code snippet
    max_len_summary = get_maxlen(summary)
    mean_len_description = get_meanlen(description)
    print("the number of Instance is: %s" % len(ids))
    print("the number of summary or description is: %s" % len(summary))
    print("the number of label is: %s" % len(labels))

    # convert list data to numpy array
    X_summary = gen_seq(summary, vocab, max_len_summary)
    X_description=gen_seq(description,vocab,int(mean_len_description))
    ids = np.array(ids)
    labels = np.array(labels)
    basename, _ = os.path.splitext(os.path.basename(file_name))

    # 保存得到的这些numpy.array矩阵
    np.save(os.path.join(outdir, '{}.ids.npy'.format(basename)), ids)
    np.save(os.path.join(outdir, '{}.summary.npy'.format(basename)), X_summary)
    np.save(os.path.join(outdir, '{}.description.npy'.format(basename)), X_description)
    np.save(os.path.join(outdir, '{}.labels.npy'.format(basename)), labels)


    # train word2vec
    # train_word2vec(summary+description)