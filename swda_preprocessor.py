#!/usr/bin/env python

import json

from collections import defaultdict
from operator import itemgetter
from swda.swda import CorpusReader

# hyperparameters
training_split = 1115
test_split = 40
pad = "PAD"

# convert unused dialogue act
# please see http://compprag.christopherpotts.net/swda.html#ex
train_label = ['aa', 'arp', 'ad', 'b^m', 'sv', 'qrr', 'ar', 'fp', '%', 'nn', 'no', 'na', 'ng', 'ny', 'qw^d', 'bd', 'qy^d', 'bf', 'ft', 'ba', 'bh', 'bk', 'fa', 'fc', 'br', 'qh', 'oo', 'b', 'qw', 'qy', 'h', 't3', 'o', 't1', '^h', 'aap', '^q', 'x', 'sd', '^2', 'qo', '^g']
convert = {
    '+': 'sd',
    'fo_o_fw_"_by_bc': 'sd',
    'oo_co_cc': 'sd',
    'arp_nd': 'no',
    'aap_am': 'sd'
}

# assuming SWDA corpus installed in path-to-project/swda
# url of repo is https://github.com/cgpotts/swda
# proprocessor script for this model is in https://github.com/miyamotost/swda
corpus = CorpusReader('swda/swda')

with open('dataset/swda_datset_training.txt', mode='a') as f1, open('dataset/swda_datset_test.txt', mode='a') as f2:
    for i, trans in enumerate(corpus.iter_transcripts(display_progress=False)):
        speakerids = [pad, pad, pad, pad]
        utts = [pad, pad, pad, pad]
        labels = [pad, pad, pad, pad]
        print('iter: {}'.format(i+1))

        #
        # speakerid   : utt.caller_no
        # main_topics : trans.topic_description しばらく"PAD"で対応（無視する）
        # pos         : utt.act_tag しばらく"PAD"で対応（無視する）
        # utt         : utt.text
        # label       : utt.act_tagm, utt.damsl_act_tag() モデルで使用されていないlabelは使用されているものに変換する
        #

        for utt in trans.utterances:
            speakerids.append(str(utt.caller_no))
            utts.append(utt.text)
            utt.act_tag = utt.damsl_act_tag()
            if (utt.act_tag not in train_label):
                if (utt.act_tag in convert.keys()):
                    utt.act_tag = convert[utt.act_tag]
                else:
                    print('Invalid act label: {}'.format(utt.act_tag))
            labels.append(utt.act_tag)
            line = json.dumps({
                'speakerid': speakerids[-4:],
                'main_topics': [pad, pad, pad, pad],
                'pos': [pad, pad, pad, pad],
                'utt': utts[-4:],
                'label': labels[-4:]
            })
            if (0 <= i < training_split):
                f1.write('{}\n'.format(line))
            if (training_split <= i < training_split+test_split):
                f2.write('{}\n'.format(line))
