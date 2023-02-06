from __future__ import absolute_import, division, print_function
from lib2to3.pgen2 import token

import logging
from io import open
from typing_extensions import ParamSpec
from regex import P
from transformers import AutoTokenizer,BartTokenizer
import pickle
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

class InputExample():
    """A single set of features of data."""
    def __init__(self,token_a,token_b,label,domain,parent=None,child=None):
        self.token_a = token_a
        self.token_b = token_b
        self.label = label
        self.domain = domain
        self.parent = parent
        self.child = child

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label,domain=None,p=None,c=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.domain = domain
        self.p = p
        self.c = c

class DataLoader__(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dirs):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dirs):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dirs):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class StanceDataloader(DataLoader__):
    def get_train_examples(self, data_dirs,topic):
        if str(topic)=='all':
            return self._create_examples(data_dirs=data_dirs, set_type='train',genre=topic)
        else: return self._create_examples_l(data_dirs=data_dirs, set_type='train',genre=topic)

    def get_dev_examples(self, data_dirs,topic):
        if str(topic)=='all':
            return self._create_examples(data_dirs=data_dirs, set_type='dev',genre=topic)
        else: return self._create_examples_l(data_dirs=data_dirs, set_type='dev',genre=topic)

    def get_test_examples(self, data_dirs,topic):
        if str(topic)=='all':
            return self._create_examples(data_dirs=data_dirs, set_type='test',genre=topic)
        else: return self._create_examples_l(data_dirs=data_dirs, set_type='test',genre=topic)

    def get_data_examples(self, data_dirs,topic):
        return self._create_examples_l(data_dirs=data_dirs, set_type='all',genre=topic)

    def _create_examples(self, data_dirs, set_type, genre='all'):
        examples = []

        data = pd.read_csv(data_dirs, index_col = False)
        data = data.sort_values(by=['datetime'])
        if genre == 'all':
            data = data
            if set_type == 'train':
                data = data.iloc[0:int(len(data)*0.8)]
            elif set_type =='dev':
                data = data.iloc[int(len(data)*0.8):int(len(data)*0.9)]
            elif set_type == 'test':
                data = data.iloc[int(len(data)*0.9):int(len(data))]
            elif set_type == 'all':
                data = data
        
        texts_a = data['body_parent']
        texts_b = data['body_child']
        labels = data['label']
        domain = data['subreddit']
        p = data['author_parent']
        c = data['author_child']
        for t in enumerate(zip(texts_a,texts_b,labels,domain,p,c)):
            t = t[1]
            examples.append(InputExample(t[0],t[1],t[2],t[3],t[4],t[5]))

        return examples

    def _create_examples_l(self, data_dirs, set_type, genre=[]):
        examples = []

        data = pd.read_csv(data_dirs, index_col = False)
        new_date = []
        data = data.sort_values(by=['datetime'])

        for gen in genre:
            data2 = data[data['subreddit']==gen]
            if set_type == 'train':
                data2 = data2.iloc[0:int(len(data2)*0.8)]
            elif set_type =='dev':
                data2 = data2.iloc[int(len(data2)*0.8):int(len(data2)*0.9)]
            elif set_type == 'test':
                data2 = data2.iloc[int(len(data2)*0.9):int(len(data2))]
        
            texts_a = data2['body_parent']
            texts_b = data2['body_child']
            labels = data2['label']
            domain = data2['subreddit']
            p = data2['author_parent']
            c = data2['author_child']

            for t in enumerate(zip(texts_a,texts_b,labels,domain,p,c)):
                t = t[1]
                examples.append(InputExample(t[0],t[1],t[2],t[3],t[4],t[5]))

        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop()

def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,unique_nodes_mapping=None,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    unique_nodes_mapping = pickle.load(open('utils/unique_nodes_n_mapping_bert.pkl', 'rb'))

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.token_a)

        tokens_b = None
        if example.token_b:
            tokens_b = tokenizer.tokenize(example.token_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # if not (len(tokens_a)+len(tokens_b)<200) and  (len(tokens_a)+len(tokens_b)>100):
        #     continue
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label=example.label,
                              domain = example.domain,
                              p=unique_nodes_mapping[example.parent],
                              c = unique_nodes_mapping[example.child]
                              ))
    return features

def convert_examples_to_features_(examples, max_seq_length,
                                 tokenizer):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    unique_nodes_mapping = pickle.load(open('utils/unique_nodes_n_mapping_bert.pkl', 'rb'))

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text = example.token_a + ' ' + str(tokenizer.sep_token) + ' ' + example.token_b

        inputs = tokenizer(text, padding='max_length', max_length=max_seq_len,truncation=True)

        assert len(inputs['input_ids']) == max_seq_len
        assert len(inputs['attention_mask']) == max_seq_len
 
        # print(example)
        features.append(
                InputFeatures(input_ids=inputs['input_ids'],
                              input_mask=inputs['attention_mask'],
                              segment_ids=None,
                              label=example.label,
                              domain = example.domain,
                              p=unique_nodes_mapping[example.parent],
                              c = unique_nodes_mapping[example.child]
                              ))
    return features

class SentGloveFeatures(object):
    def __init__(self,tokens,  embeddings,input_mask,label,domain = None):
        self.text_a = tokens
        self.input_mask = input_mask
        self.label = label
        self.domain = domain
        self.embeddings = embeddings

def get_all_words(examples,GLOVE_DIR):
    all_words=set()
    embeddings_index = {}
    for (ex_index, example) in enumerate(examples):
        tokens_a = word_tokenize(example.token_a)
        tokens_b = word_tokenize(example.token_b)
        for a in tokens_a:
            all_words.add(a)
        for b in tokens_b:
            all_words.add(b)

    with open('all_words.txt','w',encoding='utf8') as fs:
        for a in all_words:
            fs.write(a+" ")
        fs.close()
    

    with open('glove.6B.300d.txt',encoding='utf8') as f:
        with open('all_vectors.txt','wb') as fs:
            for line in f:
                values = line.split()
                word = values[0]
                if word in all_words:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                    # fs.write(word+' '+str(coefs)+'\n')
                if word == 'unk':
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index['unk'] = coefs
                    # fs.write('unk'+' '+str(coefs)+'\n')
            pickle.dump(embeddings_index,fs)
            print('Finished')

def retrive_word_embedding(examples,  max_seq_length,
                           pad_on_left=False, pad_token='unk',
                            mask_padding_with_zero=True):
    features = []
    all_words_vectors={}
    with open('all_vectors.txt','rb') as f:
        all_words_vectors = pickle.load(f)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = word_tokenize(example.token_a)
        if len(tokens_a) > max_seq_length:
                tokens_a = tokens_a[:(max_seq_length)]
        input_mask = [1 if mask_padding_with_zero else 0] * len(tokens_a)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(tokens_a)
        words = tokens_a + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        seq_vectors = [(all_words_vectors[a] if a in all_words_vectors.keys() else all_words_vectors['unk']) for a in words]
        seq_vectors = np.array(seq_vectors)
        features.append(SentGloveFeatures(tokens_a,seq_vectors,input_mask,label=example.label, domain=example.domain))
    return features

def get_stance_dataset(domain = 'labeled_data',exp_type='train'):
    data = pd.read_csv(domain+".csv", index_col = False)
    data = data.sort_values(by=['datetime'])
    nodes = []
    if exp_type == 'train':
        data = data.iloc[0:int(len(data)*0.8)]
    elif exp_type =='dev':
        data = data.iloc[int(len(data)*0.8):int(len(data)*0.9)]
    elif exp_type == 'test':
        data = data.iloc[int(len(data)*0.9):int(len(data))]
    parent = (data['author_parent'])
    child = (data['author_child'])
    for p in enumerate(zip(parent,child)):
        p=p[1]
        nodes.append((p[0],p[1]))
    return nodes

