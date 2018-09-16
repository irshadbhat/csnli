#!/usr/bin/env python

from __future__ import division, unicode_literals

import re
import os
import math
import codecs
import string
import argparse
from itertools import count
from StringIO import StringIO

import opts
import onmt
import torch
import onmt.io
import onmt.modules
import onmt.translate
import onmt.ModelConstructor

from jamo import h2j
from wxconv import WXC

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()

if opt.lang == 'hin':
    to_wx = WXC(order='utf2wx', lang='hin')
    non_alpha = re.compile(u'([^a-zA-Z]+)')
    alpha_letters = set(string.ascii_letters)

trans_dict = dict()
broken_words = dict()

def addone(text):
    for word in text.split():
        if opt.lang == 'hin' and (word[0] == word[-1] == '_'):
            trans_dict[word] = word[1:-1]
            continue
        if word in trans_dict:
            continue
        words = non_alpha.split(word)
        if len(words) == 1:
            if words[0][0] in alpha_letters:
                yield words[0]
            else:
                trans_dict[words[0]] = words[0]
        else:
            broken_words[word] = [w for w in words if w]
            for w in words:
                if not w: continue
                if w[0] in alpha_letters:
                    yield w
                else:
                    trans_dict[w] = w

def main():
    global broken_words
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    for line in codecs.open(opt.src, 'r', 'utf-8'):
        if opt.lang == 'hin':
            line = to_wx.convert(line)
        src = [' '.join(word) for word in addone(line)]

        if src:
            # Test data
            data = onmt.io.build_dataset(fields, opt.data_type,
                                         src, opt.tgt,
                                         src_dir=opt.src_dir,
                                         sample_rate=opt.sample_rate,
                                         window_size=opt.window_size,
                                         window_stride=opt.window_stride,
                                         window=opt.window,
                                         use_filter_pred=False)

            # Sort batch by decreasing lengths of sentence required by pytorch.
            # sort=False means "Use dataset's sortkey instead of iterator's".
            data_iter = onmt.io.OrderedIterator(
                dataset=data, device=opt.gpu,
                batch_size=opt.batch_size, train=False, sort=False,
                sort_within_batch=True, shuffle=False)

            # Translator
            scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta,opt.coverage_penalty, opt.length_penalty)
            translator = onmt.translate.Translator(model, fields,
                                                   beam_size=opt.beam_size,
                                                   n_best=opt.n_best,
                                                   global_scorer=scorer,
                                                   max_length=opt.max_length,
                                                   copy_attn=model_opt.copy_attn,
                                                   cuda=opt.cuda,
                                                   beam_trace=opt.dump_beam != "",
                                                   stepwise_penalty=opt.stepwise_penalty,	
                                                   min_length=opt.min_length)
            builder = onmt.translate.TranslationBuilder(
                data, translator.fields,
                opt.n_best, opt.replace_unk, opt.tgt)

            for batch in data_iter:
                batch_data = translator.translate_batch(batch, data)
                translations = builder.from_batch(batch_data)

                for trans in translations:
                    trans_dict[''.join(trans.src_raw)]  = ''.join(trans.pred_sents[0])
       
        if broken_words: 
            for w,t in broken_words.items():
                tr = ''
                for ti in t:
                    tr += trans_dict[ti]
                trans_dict[w] = tr
        broken_words = dict()
        trans_text = ' '.join([trans_dict[w] for w in line.split()])
        print trans_text


if __name__ == "__main__":
    main()
