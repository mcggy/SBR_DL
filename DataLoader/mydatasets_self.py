import re
import os
from torchtext import data
import random
import torch
import dill
import pandas as pd
from DataProcess.parse import preprocess_br,dealNan
# from DataUtils.Common import seed_num
torch.manual_seed(233)
random.seed(233)


class MyDataset(data.Dataset):

    def __init__(self, text_field, label_field, data_df=None, examples=None, char_data=None, **kwargs):
        """
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            char_data: The char level to solve
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string=dealNan(string)
            dealed_string=preprocess_br(string)
            return ' '.join(dealed_string)

        # text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            examples = []

            # ids = list(range(0, df.iloc[:, 0].size))
            # text = list(data_df['bug_title_pro']+data_df['bug_description_pro'])
            text = list(data_df['bug_title_pro'])
            # description = list(df['bug_description'])
            labels = list(data_df['Security'])
            negative, positive = 0, 0
            for i in range(len(labels)):
                if char_data is True:
                    text[i] = text[i].split(" ")
                    text[i] = MyDataset.char_data(self, text[i])
                # summary[i] = clean_str(summary[i])
                if labels[i] == 0:
                    negative += 1
                    examples += [data.Example.fromlist([text[i], 'negative'], fields=fields)]
                elif labels[i] == 1:
                    positive += 1
                    examples += [data.Example.fromlist([text[i], 'positive'], fields=fields)]

            print("a {} b {} ".format(negative, positive))
        super(MyDataset, self).__init__(examples, fields, **kwargs)

    def char_data(self, list):
        data = []
        for i in range(len(list)):
            for j in range(len(list[i])):
                data += list[i][j]
        return data

    @classmethod
    def splits(cls, all_data, split, char_data, text_field, label_field,iter_path, dev_ratio=.1, shuffle=True ,root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if not os.path.exists(iter_path):
            split_index_1 = int(len(all_data) * split[0])
            split_index_2 = int(len(all_data) * (split[0] + split[1]))
            split_index_3 = int(len(all_data) * (split[0] + split[1] + split[2]))

            train_df = all_data[0:split_index_1]
            dev_df = all_data[split_index_1:split_index_2]
            test_df = all_data[split_index_2:split_index_3]

            examples_train = cls(text_field, label_field, train_df, char_data=char_data, **kwargs).examples
            examples_dev = cls(text_field, label_field, dev_df, char_data=char_data, **kwargs).examples
            examples_test = cls(text_field, label_field, test_df, char_data=char_data, **kwargs).examples
            with open(iter_path, 'wb') as f:
                dill.dump([examples_train,examples_dev,examples_test], f)
        else:
            with open(iter_path, 'rb') as f:
                examples_train,examples_dev,examples_test = dill.load(f)


        if shuffle:
            print("shuffle data examples......")
            random.shuffle(examples_train)
            random.shuffle(examples_dev)
            random.shuffle(examples_test)

        return (cls(text_field, label_field, examples=examples_train),
                cls(text_field, label_field, examples=examples_dev),
                cls(text_field, label_field, examples=examples_test))