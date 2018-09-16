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

from wxconv import WXC

import warnings

warnings.filterwarnings('ignore')

class Transliterate:
    def __init__(self, model, lang, gpu=False, wx=False):
        self.lang = lang
        self.is_ip_wx = wx
        parser = argparse.ArgumentParser(
            description='transliterate.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        opts.add_md_help_argument(parser)
        opts.translate_opts(parser)
        
        self.opt = parser.parse_args()
        self.trans_dict = dict()
        self.broken_words = dict()
        file_path = os.path.dirname(os.path.abspath(__file__))

        if self.lang == 'hin':
            self.to_utf = WXC(order='wx2utf', lang='hin')
            self.non_alpha = re.compile(u'([^a-zA-Z]+)')
            self.alpha_letters = set(string.ascii_letters)
            self.com_abbr = {'b':['BI', 'be'], 'd': ['xI', 'xe'], 'g':['jI'], 'k':['ke', 'ki', 'kI'],
                    'h':['hE', 'hEM'], 'ha':['hE', 'hEM'], 'n': ['ina', 'ne'], 'm':['meM', 'mEM'], 'p':['pe'],
                    'q': ['kyoM'], 'r': ['Ora', 'ora'], 's':['isa', 'se'], 'y':['ye']}

        if self.lang == 'eng':
            self.non_alpha = re.compile(u'([^a-z]+)')
            self.alpha_letters = set(string.ascii_letters[:26])
            with open('%s/extras/COMMON_ABBR.eng' %file_path) as fp:
               self.com_abbr = {}
               for line in fp:
                   k,v = line.split()
                   self.com_abbr[k] = v.split('|')
        
        dummy_parser = argparse.ArgumentParser(description='train.py')
        opts.model_opts(dummy_parser)
        dummy_opt = dummy_parser.parse_known_args([])[0]
        if gpu:
            self.opt.gpu = 0
    
        self.opt.cuda = self.opt.gpu > -1
        self.opt.model = model
        self.opt.n_best = 5
        self.opt.lang = lang
        if self.opt.cuda:
            torch.cuda.set_device(self.opt.gpu)

        # Load the model.
        self.fields, self.model, self.model_opt = onmt.ModelConstructor.load_test_model(self.opt, dummy_opt.__dict__)

    def is_url(self, word):
        if word.startswith('http') or word.startswith('www.'):
            return True
        if any(word.endswith(x) for x in '.com .org .net .int .edu .gov .mil .in .us .uk'.split()):
            return True
        return False

    def addone(self, text):
        for word in set(text.split()):
            if word[0] in '@#':
                self.trans_dict[word] = [word]*self.opt.n_best
                continue
            elif self.is_url(word):
                self.trans_dict[word] = [word]*self.opt.n_best
                continue
            if self.opt.lang == 'hin' and len(word) > 2 and (word[0] == word[-1] == '_'):
                self.trans_dict[word] = [word[1:-1]]*self.opt.n_best
                continue
            if word in self.trans_dict:
                continue
            words = self.non_alpha.split(word)
            if len(words) == 1:
                if words[0][0] in self.alpha_letters:
                    yield words[0]
                else:
                    self.trans_dict[words[0]] = [words[0]]*self.opt.n_best
            else:
                self.broken_words[word] = [w for w in words if w]
                for w in words:
                    if not w: continue
                    if w[0] in self.alpha_letters:
                        yield w
                    else:
                        self.trans_dict[w] = [w]*self.opt.n_best

    def transliterate(self, text):
        if not self.is_ip_wx:
            text = re.sub(r'([a-z])\1\1+', r'\1', text.lower())
        src = [' '.join(word) for word in self.addone(text)]
        if src:
            # Test data
            data = onmt.io.build_dataset(self.fields, self.opt.data_type,
                                         src, self.opt.tgt,
                                         src_dir=self.opt.src_dir,
                                         sample_rate=self.opt.sample_rate,
                                         window_size=self.opt.window_size,
                                         window_stride=self.opt.window_stride,
                                         window=self.opt.window,
                                         use_filter_pred=False)

            # Sort batch by decreasing lengths of sentence required by pytorch.
            # sort=False means "Use dataset's sortkey instead of iterator's".
            data_iter = onmt.io.OrderedIterator(
                dataset=data, device=self.opt.gpu,
                batch_size=self.opt.batch_size, train=False, sort=False,
                sort_within_batch=True, shuffle=False)

            # Translator
            scorer = onmt.translate.GNMTGlobalScorer(self.opt.alpha, self.opt.beta, self.opt.coverage_penalty, self.opt.length_penalty)
            translator = onmt.translate.Translator(self.model, self.fields,
                                                   beam_size=self.opt.beam_size,
                                                   n_best=self.opt.n_best,
                                                   global_scorer=scorer,
                                                   max_length=self.opt.max_length,
                                                   copy_attn=self.model_opt.copy_attn,
                                                   cuda=self.opt.cuda,
                                                   beam_trace=self.opt.dump_beam != "",
                                                   stepwise_penalty=self.opt.stepwise_penalty,	
                                                   min_length=self.opt.min_length)
            builder = onmt.translate.TranslationBuilder(
                data, translator.fields,
                self.opt.n_best, self.opt.replace_unk, self.opt.tgt)

            for batch in data_iter:
                batch_data = translator.translate_batch(batch, data)
                translations = builder.from_batch(batch_data)

                for trans in translations:
                    src_raw = ''.join(trans.src_raw)
                    self.trans_dict[src_raw]  = self.com_abbr.get(src_raw, []) + [''.join(tw) if ''.join(tw) else src_raw for tw in trans.pred_sents[:5] ]
       
        if self.broken_words: 
            for w,t in self.broken_words.items():
                tr = ''
                for ti in t:
                    tr += self.trans_dict[ti][0]
                self.trans_dict[w] = [tr]*5
        self.broken_words = dict()
        trans_text = ' '.join([self.trans_dict[w][0] for w in text.split()])
        return trans_text

