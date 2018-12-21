from __future__ import unicode_literals

import dynet_config
dynet_config.set(random_seed=127, autobatch=1)

import io
import re
import sys
sys.path.append('nmt')

import json
import pickle
import random
import enchant
from argparse import ArgumentParser

import kenlm
import enchant
import numpy as np
import dynet as dy

from wxconv import WXC
from so_viterbi import so_viterbi
from lang_tagger import *
from nmt.transliterate import Transliterate


class ThreeStepDecoding(object):
    def __init__(self, lid, htrans=None, etrans=None, wx=False):
        self.ed = enchant.Dict('en')
        self.hblm =  kenlm.LanguageModel('lm/hindi-n3-p5-lmplz.blm')
        self.eblm =  kenlm.LanguageModel('lm/english-n3-p10-lmplz.blm')
        self.so_dec_eng = so_viterbi(self.eblm)
        self.so_dec_hin = so_viterbi(self.hblm)
        self.e2h = {kv.split()[0]:kv.split()[1].split('|') for kv in io.open('dicts/ENG2HIN12M.dict')}
        self.h2e = {kv.split()[0]:kv.split()[1].split('|') for kv in io.open('dicts/HIN2ENG12M.dict')}
        self.meta = Meta()
        self.lid = LID(model=lid, etrans=etrans, htrans=htrans)
        self.wx = wx
        if not self.wx:
            self.wxc = WXC(order='wx2utf', lang='hin')

    def max_likelihood(self, n_sentence, target, k_best=7):
        if target == 'en':
            auto_tags = self.so_dec_eng.decode(n_sentence, len(n_sentence), k_best)
        else:
            auto_tags = self.so_dec_hin.decode(n_sentence, len(n_sentence), k_best)
        #beamsearch
        best_sen = [n_sentence[idx][at] for idx, at in enumerate(auto_tags)]
        return best_sen

    def decode(self, words, ltags):
        words = [re.sub(r'([a-z])\1\1+', r'\1', w) for w in words]
        hi_trellis = [self.lid.htrans.get(wi.lower(), [wi]*5)[:5] + 
                      self.e2h.get(wi.lower() if li == 'en' else None, [u'_%s_' %wi])[:1] + 
                      [u'_%s_'%wi] for wi,li in zip(words, ltags)]
        hi_mono = self.max_likelihood(hi_trellis, 'hi')
        en_trellis = [[wi] + self.lid.etrans.get(wi.lower(), [wi]*5)[:5] + 
                      self.h2e.get(wh if li == 'hi' else None, [wi])[:1] 
                      for wi,wh,li in zip (words, hi_mono, ltags)]
        en_mono = self.max_likelihood(en_trellis, 'en')
        out = hi_mono[:]
        for i, _ in enumerate(hi_mono):
            if ltags[i]  in ['univ', 'acro', 'ne']:
                out[i] = words[i]
            elif ltags[i] in ['en', 'ne']:
                if words[i].lower() == en_mono[i]:
                    out[i] = words[i]
                elif self.ed.check(words[i].lower()) and len(words[i])>1:
                    out[i] = words[i]
                elif words[i].lower() in ['a', 'i']:
                    out[i] = words[i]
                else:
                    out[i] = en_mono[i]
            elif not self.wx:
                out[i] = self.wxc.convert(out[i])
        return out
    
    def tag_sent(self, sent, trans=True):
        sent = sent.split()
        sent, ltags = zip(*self.lid.tag_sent(sent))
        dec = self.decode(sent, ltags)
        return zip(sent, dec, ltags)
    
if __name__ == '__main__':
    parser = ArgumentParser(description="Language Identification System")
    parser.add_argument('--test-file', dest='tfile', required=True, help='Raw Test file')
    parser.add_argument('--lid-model', dest='lid_model', help='Load Pretrained Model')
    parser.add_argument('--etrans', required=True, help='OpenNMT English Transliteration Model')
    parser.add_argument('--htrans', required=True, help='OpenNMT Hindi Transliteration Model')
    parser.add_argument('--wx', action='store_true', help='set this flag to return Hindi words in WX')
    parser.add_argument('--output-file', dest='ofile', default='/tmp/out.txt', help='Output File')
    args = parser.parse_args()

    tsd = ThreeStepDecoding(args.lid_model, args.htrans, args.etrans, wx=args.wx)
    tsd.lid.en_trans.transliterate('\n'.join(set(io.open(args.tfile).read().split())))
    tsd.lid.etrans = tsd.lid.en_trans.trans_dict
    tsd.lid.hi_trans.transliterate('\n'.join(set(io.open(args.tfile).read().split())))
    tsd.lid.htrans = tsd.lid.hi_trans.trans_dict
    with io.open(args.tfile) as ifp, io.open(args.ofile, 'w') as ofp:
        for i,sent in enumerate(ifp):
            dec_sent = tsd.tag_sent(sent, trans=False)
            ofp.write('\n'.join(['\t'.join(x) for x in dec_sent])+'\n\n')
