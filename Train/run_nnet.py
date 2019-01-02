'''
author:jingyuan
data:2018/5/10
function: run model
'''
import os
import argparse
import datetime
from Config import config as configurable, config
import torchtext.data as data
import sys
import numpy as np

sys.path.insert(0, '../')
# from batch_generator import batch_gen
from Models.model_LSTM import LSTM
from Models.model_CNN import CNN_Text
from Models.model_CNN_BiGRU import CNN_BiGRU
from Models.model_GRU import GRU
from Models.model_DeepCNN import DEEP_CNN
import torch
from DataLoader import mydatasets_self
from Load_Pretrained_Embed import load_pretrained_emb_avg
import random
import pandas as pd

from DataUtils.Common import seed_num, pad, unk
import train_ALL_LSTM,train_ALL_CNN

torch.manual_seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)
torch.cuda.manual_seed(seed_num)


def define_dict():
    """
     use torchtext to define word and label dict
    """
    print("use torchtext to define word dict......")
    config.text_field = data.Field(lower=True)
    config.label_field = data.Field(sequential=False)
    config.static_text_field = data.Field(lower=True)
    config.static_label_field = data.Field(sequential=False)
    print("use torchtext to define word dict finished.")


# def combine_data(data, label):
#     deal_data = np.dstack((data, label))
#     deal_data = deal_data.reshape(-1, 2)
#     return deal_data


def Load_Data():
    """
    load five classification task data and two classification task data
    :return:
    """
    train_iter, dev_iter, test_iter = None, None, None
    print("Executing 2 Classification Task......")

    # the proportion of training data, validate data ,testing data
    split = [0.5, 0.25, 0.25]

    data_df = pd.read_csv(config.file_path, encoding='iso-8859-1')
    data_df.sort_values(by='issue_id', inplace=True)
    train_iter, dev_iter, test_iter = produce_iter(data_df, split, config.char_data, config.text_field,
                                                   config.label_field, device=-1, repeat=False,
                                                   shuffle=config.epochs_shuffle, sort=False)

    return train_iter, dev_iter, test_iter


def produce_iter(all_data, split, char_data, text_field, label_field, **kargs):
    """
    :function: load two-classification data
    :param path:
    :param train_name: train path
    :param dev_name: dev path
    :param test_name: test path
    :param char_data: char data
    :param text_field: text dict for finetune
    :param label_field: label dict for finetune
    :param kargs: others arguments
    :return: batch train, batch dev, batch test
    """
    train_data, dev_data, test_data = mydatasets_self.MyDataset.splits(all_data, split, char_data, text_field,
                                                                       label_field, config.iter_path)
    print("len(train_data) {} ".format(len(train_data)))
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    text_field.build_vocab(train_data.text, min_freq=config.min_freq)
    # text_field.build_vocab(train_data.text, dev_data.text, test_data.text, min_freq=config.min_freq)
    label_field.build_vocab(train_data.label)
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data), batch_sizes=(
        config.batch_size, config.batch_size, config.batch_size), **kargs)
    return train_iter, dev_iter, test_iter


def update_arguments():
    config.lr = config.learning_rate
    config.embed_num = len(config.text_field.vocab)
    config.class_num = len(config.label_field.vocab) - 1
    config.init_weight_decay = config.weight_decay
    config.init_clip_max_norm = config.clip_max_norm
    config.paddingId = config.text_field.vocab.stoi[pad]
    config.unkId = config.text_field.vocab.stoi[unk]
    if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
        config.embed_num_mui = len(config.static_text_field.vocab)
        config.paddingId_mui = config.static_text_field.vocab.stoi[pad]
        config.unkId_mui = config.static_text_field.vocab.stoi[unk]
    # config.kernel_sizes = [int(k) for k in config.kernel_sizes.split(',')]
    print(config.kernel_sizes)
    mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.mulu = mulu
    config.save_dir = os.path.join("" + config.save_dir, config.mulu)
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)


# def test(test_iter, model):
#     y_true = []
#     y_pred = []
#     for batch in test_iter:
#         feature, target = batch.text, batch.label.data.sub_(1)
#         if config.cuda is True:
#             feature, target = feature.cuda(), target.cuda()
#         y_true.extend(target.cpu().numpy().tolist())
#         out = model(feature)
#         y_pred.extend(torch.max(out, 1)[1].cpu().data.numpy().tolist())
#     return y_true, y_pred


def load_preEmbedding():
    # load word2vec
    static_pretrain_embed = None
    pretrain_embed = None
    if config.word_Embedding:
        print("word_Embedding_Path {} ".format(config.word_Embedding_Path))
        path = config.word_Embedding_Path
        print("loading pretrain embedding......")
        paddingkey = pad
        pretrain_embed = load_pretrained_emb_avg(path=path, text_field_words_dict=config.text_field.vocab.itos,
                                                 pad=paddingkey)
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            static_pretrain_embed = load_pretrained_emb_avg(path=path,
                                                            text_field_words_dict=config.static_text_field.vocab.itos,
                                                            pad=paddingkey)
        config.pretrained_weight = pretrain_embed
        if config.CNN_MUI is True or config.DEEP_CNN_MUI is True:
            config.pretrained_weight_static = static_pretrain_embed

        print("pretrain embedding load finished!")


# train model
def start_train():
    train_iter, valid_iter, test_iter = Load_Data()
    # load pretrain embedding
    load_preEmbedding()

    update_arguments()
    model = CNN_Text(config)
    if config.cuda is True:
        model = model.cuda()
    train_ALL_CNN.train(train_iter, valid_iter, test_iter, model, config)


if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument('--config_file', default="../Config/config.cfg")
    config = parser.parse_args()

    config = configurable.Configurable(config_file=config.config_file)

    if config.cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())

    # train_model('chromium')
    define_dict()
    start_train()

    # y_pred=np.argmax(lstm_model().data.numpy(), axis=1)
    # # y_pred = np.where(y_pred < 0.5, 0, 1)
    # result = evaluate_b()
    # print(result)
