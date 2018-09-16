from __future__ import unicode_literals

import dynet_config
dynet_config.set(random_seed=127, autobatch=1)

import io
import re
import sys
sys.path.append('nmt')

import pickle
import random
import enchant
from argparse import ArgumentParser

import numpy as np
import dynet as dy
from gensim.models.word2vec import Word2Vec

from nmt.transliterate import Transliterate

htrans, etrans = dict(), dict()

class Meta:
    def __init__(self):
        self.c_dim = 32  # Character Embeddings Input dimnesions
        self.f_dim = 8  # Flag Dimensions
        self.n_flags = 12  # Number of Flag types
        self.w_dim_t = 0  # word embedding size
        self.w_dim_en = 0  # pretrained word embedding size (0 if no pretrained embeddings)
        self.w_dim_hi = 0  # pretrained word embedding size (0 if no pretrained embeddings)
        self.add_words_t = 1  # additional lookup for missing/special words
        self.add_words_en = 1  # additional lookup for missing/special words
        self.add_words_hi = 1  # additional lookup for missing/special words
        self.n_hidden = 64  # No. of Hidden Units in MLP
        self.lstm_flag_dim = 24
        self.lstm_char_dim = 64  # LSTM Character dimnesions
        self.lstm_word_dim = 64  # LSTM Word dimnesions


class LID(object):
    def __init__(self, model=None, meta=None, etrans=None, htrans=None):
        self.model = dy.Model()
        self.ed = enchant.Dict('en')
        self.meta = pickle.load(open('%s.meta' %model, 'rb')) if model else meta
        self.trainer = self.meta.trainer(self.model)
        if self.meta.w_dim_en:
            self.WORDS_LOOKUP_EN = self.model.add_lookup_parameters((self.meta.n_words_en, self.meta.w_dim_en))
            if not model:
                for word, V in ewvm.vocab.iteritems():
                    self.WORDS_LOOKUP_EN.init_row(V.index+self.meta.add_words_en, ewvm.syn0[V.index])
        if self.meta.w_dim_hi:
            self.WORDS_LOOKUP_HI = self.model.add_lookup_parameters((self.meta.n_words_hi, self.meta.w_dim_hi))
            if not model:
                for word, V in hwvm.vocab.iteritems():
                    self.WORDS_LOOKUP_HI.init_row(V.index+self.meta.add_words_hi, hwvm.syn0[V.index])

        if self.meta.w_dim_t:
            self.WORDS_LOOKUP_T = self.model.add_lookup_parameters((self.meta.n_words_t, self.meta.w_dim_t))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_chars, self.meta.c_dim))
        self.FLAGS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_flags, self.meta.f_dim))

        # MLP on top of biLSTM outputs 
        self.W1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_word_dim*2))
        self.W2 = self.model.add_parameters((self.meta.n_tags, self.meta.n_hidden))
        self.B1 = self.model.add_parameters(self.meta.n_hidden)
        self.B2 = self.model.add_parameters(self.meta.n_tags)

        # word-level LSTMs
        lstm_ip_dim = self.meta.w_dim_en+self.meta.w_dim_hi+self.meta.w_dim_t+self.meta.lstm_char_dim*2+self.meta.lstm_flag_dim
        self.fwdRNN = dy.LSTMBuilder(1, lstm_ip_dim, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN = dy.LSTMBuilder(1, lstm_ip_dim, self.meta.lstm_word_dim, self.model)

        # char-level LSTMs
        self.cfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)
        self.cbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.lstm_char_dim, self.model)

        self.en_trans = None
        self.hi_trans = None
        self.htrans = dict()
        self.etrans = dict()
        if etrans:
            self.load_etrans(etrans)
        if htrans:
            self.load_htrans(htrans)

        if model:
            self.model.populate('%s.dy' %model)

    def load_etrans(self, etrans, train=None, dev=None, test=None):
        self.en_trans = Transliterate(etrans, lang='eng')
        if train:
            self.en_trans.transliterate('\n'.join(set([w for sent in train for w,t in sent])))
        if dev:
            self.en_trans.transliterate('\n'.join(set([w for sent in dev for w,t in sent])))
        if test:
            self.en_trans.transliterate('\n'.join(set(sum(test, []))))
        self.etrans = self.en_trans.trans_dict

    def load_htrans(self, htrans, train=None, dev=None, test=None):
        self.hi_trans = Transliterate(htrans, lang='hin')
        if train:
            self.hi_trans.transliterate('\n'.join(set([w for sent in train for w,t in sent])))
        if dev:
            self.hi_trans.transliterate('\n'.join(set([w for sent in dev for w,t in sent])))
        if test:
            self.hi_trans.transliterate('\n'.join(set(sum(test, []))))
        self.htrans = self.hi_trans.trans_dict

    def set_flags(self, x):
        vec = []
        lng = len(x)
        if self.ed.check(x.lower()):
            vec.append(0)
        else:
            vec.append(1)
        if self.ed.check(x.title()):
            vec.append(2)
        else:
            vec.append(3)
        if 1 <= lng <= 3:
            vec.append(8)
        elif 4 <= lng <= 5:
            vec.append(9)
        elif 6 <= lng <= 8:
            vec.append(10)
        else:
            vec.append(11)
        return vec


    def get_e_index(self, ow):
        try:
            return self.meta.ew2i[ow]
            return self.meta.ew2i[ow.lower()]
        except KeyError:
            pass
        for w in self.etrans.get(ow.lower(), []):
            try:
                return self.meta.ew2i[w]
            except KeyError:
                pass
        return 0
    
    def get_h_index(self, ow):
        for w in self.htrans.get(ow.lower(), []):
            try:
                return self.meta.hw2i[w]
            except KeyError:
                pass
        return 0

    def word_rep(self, w):
        w = re.sub(r'([a-z])\1\1+', r'\1', w)
        embd = []
        if self.meta.w_dim_en:
            drop_word = not self.eval and random.random() < 0.5
            e_idx = 0 if drop_word else self.get_e_index(w)
            embd.append(self.WORDS_LOOKUP_EN[e_idx])
        if self.meta.w_dim_hi:
            drop_word = not self.eval and random.random() < 0.5
            h_idx = 0 if drop_word else self.get_h_index(w)
            embd.append(self.WORDS_LOOKUP_HI[h_idx])
        if not embd:
            drop_word = not self.eval and random.random() < 0.5
            t_idx = 0 if drop_word else self.meta.tw2i.get(w, 0)
            return self.WORDS_LOOKUP_T[t_idx] 
        return dy.concatenate(embd) if len(embd) > 1 else embd[0]
    
    def char_rep(self, w, f, b):
        char_drop = not self.eval and random.random() < 0.1
        bos, eos, unk = self.meta.c2i["bos"], self.meta.c2i["eos"], self.meta.c2i["unk"]
        char_ids = [bos] + [unk if char_drop else self.meta.c2i.get(c, unk) for c in w] + [eos]
        char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        cemb = dy.concatenate([ fw_exps[-1], bw_exps[-1] ])
        return cemb

    def enable_dropout(self):
        self.w1 = dy.dropout(self.w1, 0.5)
        self.b1 = dy.dropout(self.b1, 0.5)
        self.fwdRNN.set_dropout(0.5)
        self.bwdRNN.set_dropout(0.5)
        self.cfwdRNN.set_dropout(0.5)
        self.cbwdRNN.set_dropout(0.5)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.cfwdRNN.disable_dropout()
        self.cbwdRNN.disable_dropout()
    
    def build_tagging_graph(self, words):
        # parameters -> expressions
        self.w1 = dy.parameter(self.W1)
        self.b1 = dy.parameter(self.B1)
        self.w2 = dy.parameter(self.W2)
        self.b2 = dy.parameter(self.B2)

        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout()

        # initialize the RNNs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()

        cf_init = self.cfwdRNN.initial_state()
        cb_init = self.cbwdRNN.initial_state()
    
        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        wembs = [self.word_rep(w) for w in words]
        cembs_raw = [self.char_rep(w, cf_init, cb_init) for w in words]
        flags = [dy.concatenate([self.FLAGS_LOOKUP[f] for f in self.set_flags(w)]) for w in words]
    
        wembs = [dy.concatenate(list(embs)) for embs in zip(wembs, cembs_raw, flags)]
    
        # feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))
    
        # biLSTM states
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

        # feed each biLSTM state to an MLP
        exps = []
        for xi in bi_exps:
            xh = self.meta.activation(self.w1 * xi) + self.b1
            xo = self.w2*xh + self.b2
            exps.append(xo)
    
        return exps
    
    def sent_loss(self, words, tags):
        self.eval = False
        vecs = self.build_tagging_graph(words)
        for v,t in zip(vecs,tags):
            tid = self.meta.t2i[t]
            err = dy.pickneglogsoftmax(v, tid)
            self.loss.append(err)
    
    def tag_sent(self, words, trans=True):
        self.eval = True
        if trans:
            if self.en_trans:
                self.en_trans.transliterate('\n'.join(set(words)))
                self.etrans = self.en_trans.trans_dict
            if self.hi_trans:
                self.hi_trans.transliterate('\n'.join(set(words)))
                self.htrans = self.hi_trans.trans_dict
        dy.renew_cg()
        vecs = self.build_tagging_graph(words)
        vecs = [dy.softmax(v) for v in vecs]
        probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in probs:
            tag = np.argmax(prb)
            tags.append(self.meta.i2t[tag])
        return zip(words, tags)

def eval_model(dev, ofp=None):
    gall, pall = [], []
    good_sent = bad_sent = good = bad = 0.0
    for sent in dev:
        words = [w for w,t in sent]
        golds = [t for w,t in sent]
        gall.extend(golds)
        tagged = lid.tag_sent(words)
        if ofp is not None:
            ofp.write('\n'.join(['\t'.join(x) for x in tagged])+'\n\n')
        tags = [t for w,t in tagged]
        pall.extend(tags)
        if tags == golds: good_sent += 1
        else: bad_sent += 1
        for go,gu in zip(golds,tags):
            if go == gu: good += 1
            else: bad += 1
    score = good/(good+bad)
    print(score, good_sent/(good_sent+bad_sent))
    return score

def train_lid(train, dev):
    def _update():
        batch_loss = dy.esum(lid.loss)
        loss = batch_loss.scalar_value()
        batch_loss.backward()
        lid.trainer.update()
        lid.loss = []
        dy.renew_cg()
        return loss
    pr_acc = 0.0
    n_samples = len(train)
    num_tagged, cum_loss = 0, 0
    for ITER in xrange(args.iter):
        dy.renew_cg()
        lid.loss = []
        random.shuffle(train)
        for i,s in enumerate(train, 1):
            if i % 500 == 0 or i == n_samples:   # print status
                lid.trainer.status()
                print(cum_loss / num_tagged)
                cum_loss, num_tagged = 0, 0
            # train on sent
            words, golds = zip(*s)
            lid.sent_loss(words, golds)
            num_tagged += len(golds)
            if len(lid.loss) > 50:
                cum_loss += _update()
        if lid.loss:
            cum_loss += _update()
        acc = eval_model(dev)
        if acc > pr_acc:
            pr_acc = acc
            print('Save Point:: %d' %ITER)
            if args.save_model:
                lid.model.save('%s.dy' %args.save_model)
        print("epoch %r finished" % ITER)
        sys.stdout.flush()

def read(fname):
    data, sent = [], []
    fp = io.open(fname, encoding='utf-8')
    for i,line in enumerate(fp):
        line = line.split()
        if not line:
            data.append(sent)
            sent = []
        else:
            try:
                w,p = line
            except ValueError:
                try:
                    w,p = line[1], line[8]
                except Exception:
                    sys.stderr.write('Wrong file format\n')
                    sys.exit(1)
            sent.append((w,p))
    if sent: data.append(sent)
    return data

def set_label_map(data):
    tags = set()
    meta.tw2i = {}
    meta.c2i = {'bos':0, 'eos':1, 'unk':2}
    cid = len(meta.c2i)
    eword = hword = ''
    for sent in data:
        for word,tag in sent:
            if not args.eembd:
                meta.tw2i.setdefault(word, 0)
                meta.tw2i[word] += 1
            tags.add(tag)
            for c in word:
                if not meta.c2i.has_key(c):
                    meta.c2i[c] = cid
                    cid += 1
    if not args.eembd:
        meta.w_dim_t = 64
        meta.tw2i = {w:i for i,w in enumerate([w for w,c in meta.tw2i.items() if c > 2], 1)}
        meta.n_words_t = len(meta.tw2i) + meta.add_words_t
    meta.n_chars = len(meta.c2i)
    meta.n_tags = len(tags)
    meta.i2t = dict(enumerate(tags))
    meta.t2i = {t:i for i,t in meta.i2t.items()}

if __name__ == '__main__':
    parser = ArgumentParser(description="Language Identification System")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-seed', dest='seed', type=int, default=127)
    parser.add_argument('--train', help='CONLL/TNT Train file')
    parser.add_argument('--dev', help='CONLL/TNT Dev/Test file')
    parser.add_argument('--test', help='Raw Test file')
    parser.add_argument('--eng-pretrained-embd', dest='eembd', help='Pretrained word2vec Embeddings')
    parser.add_argument('--hin-pretrained-embd', dest='hembd', help='Pretrained word2vec Embeddings')
    parser.add_argument('--elimit', type=int, default=None, help='load top-n English word vectors (default=all vectors, recommended=400k)')
    parser.add_argument('--hlimit', type=int, default=None, help='load top-n Hindi word vectors (default=all vectors, recommended=200k)')
    parser.add_argument('--trainer', default='momsgd', help='Trainer [cysgd|momsgd|adam|adadelta|adagrad|amsgrad]')
    parser.add_argument('--activation-fn', dest='act_fn', default='tanh', help='Activation function [tanh|relu|sigmoid]')
    parser.add_argument('--iter', type=int, default=100, help='No. of Epochs')
    parser.add_argument('--bvec', type=int, help='1 if binary embedding file else 0')
    parser.add_argument('--etrans', help='OpenNMT English Transliteration Model')
    parser.add_argument('--htrans', help='OpenNMT Hindi Transliteration Model')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    parser.add_argument('--output-file', dest='ofile', default='/tmp/out.txt', help='Output File')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    meta = Meta()
    dev, test = [], []
    if args.dev:
        dev = read(args.dev)
    if args.test:
        with io.open(args.test) as fp:
            test = [line.split() for line in fp]
    if not args.load_model:    
        train = read(args.train)
        if args.eembd:
            meta.ew2i = {}
            ewvm = Word2Vec.load_word2vec_format(args.eembd, binary=args.bvec, limit=args.elimit)
            meta.w_dim_en = ewvm.syn0.shape[1]
            meta.n_words_en = ewvm.syn0.shape[0]+meta.add_words_en
            for w in ewvm.vocab:
                meta.ew2i[w] = ewvm.vocab[w].index + meta.add_words_en

        if args.hembd and args.htrans:
            meta.hw2i = {}
            hwvm = Word2Vec.load_word2vec_format(args.hembd, binary=args.bvec, limit=args.hlimit)
            meta.w_dim_hi = hwvm.syn0.shape[1]
            meta.n_words_hi = hwvm.syn0.shape[0]+meta.add_words_hi
            for w in hwvm.vocab:
                meta.hw2i[w] = hwvm.vocab[w].index + meta.add_words_hi
        
        trainers = {
            'simsgd'   : dy.SimpleSGDTrainer,
            'cysgd'    : dy.CyclicalSGDTrainer,
            'momsgd'   : dy.MomentumSGDTrainer,
            'adam'     : dy.AdamTrainer,
            'adagrad'  : dy.AdagradTrainer,
            'adadelta' : dy.AdadeltaTrainer,
            'amsgrad'  : dy.AmsgradTrainer
            }
        act_fn = {
            'sigmoid' : dy.logistic,
            'tanh'    : dy.tanh,
            'relu'    : dy.rectify,
            }
        meta.trainer = trainers[args.trainer]
        meta.activation = act_fn[args.act_fn] 
        set_label_map(train)
        lid = LID(meta=meta)
        if args.etrans:
            lid.load_etrans(args.etrans, train, dev)
        if args.htrans:
            lid.load_htrans(args.htrans, train, dev)

        if args.save_model:
            pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))

        train_lid(train, dev)

    if args.load_model:
        lid = LID(model=args.load_model)
        if args.etrans:
            lid.load_etrans(args.etrans, dev=dev, test=test)
        if args.htrans:
            lid.load_htrans(args.htrans, dev=dev, test=test)
        if args.dev:
            with io.open(args.ofile, 'w') as ofp:
                eval_model(dev, ofp)
        elif args.test:
            with io.open(args.ofile, 'w') as ofp:
                for sent in test:
                    tagged = lid.tag_sent(sent, trans=False)
                    ofp.write('\n'.join(['\t'.join(x) for x in tagged])+'\n\n')

