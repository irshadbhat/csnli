#!/usr/bin/env python -*- coding: utf-8 -*- 

import cython 
import numpy as np
import kenlm as klm

cimport numpy as np
from cpython cimport bool
from cpython cimport array

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class so_viterbi:

    cdef object model

    def __init__(self, model):
        self.model = model

    def decode(self, list symbols, np.uint16_t T, np.uint16_t N):
        cdef object gen
        cdef bool missing, _missing, __missing
        cdef unicode symbol_j, symbol_k, ngram 
        cdef np.ndarray[ndim=1, dtype=np.uint16_t, mode='c'] path
        cdef np.ndarray[ndim=3, dtype=np.float64_t, mode='c'] score
        cdef np.ndarray[ndim=3, dtype=np.uint16_t, mode='c'] backtrack
        cdef np.uint16_t i, j, k, max_ind, PenUltimateState, UltimateState
        cdef np.float64_t max_val, par_score, gen_score, NEGINF = -np.inf, panelty = -100.0

        score = np.zeros((T+1,N,N)) + NEGINF

        #Initail unigram
        for i in range(N):
            #if symbols[0][i] in self.model: 
            score[0, :,i] = next(self.model.full_scores(symbols[0][i]))[0]
            #else:
            #    score[0, :,i] = next(self.model.full_scores(symbols[0][i]))[0] + panelty

        #Initail bigram
        for j in range(N):
            #missing = False
            symbol_j =  symbols[0][j]
            #if not symbol_j in self.model: 
            #    missing = True
            for k in range(N):
                #_missing = False
                #if not symbols[1][k] in self.model: 
                #    _missing = True
                ngram =  ' '.join((symbol_j, symbols[1][k]))
                gen = self.model.full_scores(ngram)
                gen_score = next(gen)[0]
                gen_score = next(gen)[0]
                #if missing:
                #    gen_score += panelty    
                #if _missing:
                #    gen_score += panelty    
                score[1, j, k] = gen_score + score[0, k, j]
                #_missing = False

        #Recursive computation for 2 <= t <= T
        backtrack = np.zeros((T+1, N, N), dtype=np.uint16)
        for t in range(2, T):
            for j in range(N):
                #missing = False
                symbol_j =  symbols[t-1][j]
                #if not symbol_j in self.model:
                #    missing = True
                for k in range(N):
                    #_missing = False
                    max_ind = 0
                    max_val = NEGINF
                    symbol_k =  symbols[t][k]
                    #if not symbol_k in self.model:
                    #    _missing = True
                    for i in range(N):
                        #__missing = False
                        #if not symbols[t-2][i] in self.model:
                        #    __missing = True
                        ngram =  ' '.join((symbols[t-2][i], symbol_j, symbol_k))
                        gen = self.model.full_scores(ngram)
                        gen_score = next(gen)[0]
                        gen_score = next(gen)[0]
                        gen_score = next(gen)[0]
                        #if missing:
                        #    gen_score += panelty
                        #if _missing:
                        #    gen_score += panelty
                        #if __missing:
                        #    gen_score += panelty
                        par_score = score[t-1, i, j] + gen_score
                        if par_score > max_val:
                            max_ind = i
                            max_val = par_score
                    score[t, j, k] = max_val
                    backtrack[t, j, k] = max_ind

        #Final bigram
        for j in range(N):
            #missing = False
            max_ind = 0
            max_val = NEGINF
            symbol_j =  symbols[T-1][j]
            #if not symbol_j in self.model:
            #    missing = True
            for i in range(N):
                #_missing = False
                #if not symbols[T-2][i] in self.model:
                #    _missing = True
                ngram =  ' '.join((symbols[T-2][i], symbol_j))
                gen = self.model.full_scores(ngram)
                gen_score = next(gen)[0]
                gen_score = next(gen)[0]
                gen_score = next(gen)[0]
                #if missing:
                #    gen_score += panelty
                #if _missing:
                #    gen_score += panelty
                par_score = gen_score + score[T-1, i, j]
                if par_score > max_val:
                    max_ind = i
                    max_val = par_score
            score[T, j] = max_val
            backtrack[T, j] = max_ind

        #Backtrack
        path = np.empty(T+1, dtype=np.uint16)
        PenUltimateState, UltimateState = np.unravel_index(np.argmax(score[T]),(N,N)) 
        path[0] = UltimateState
        path[1] = PenUltimateState
        for i,t in enumerate(range(T-2, -1, -1)):
            path[i+2] = backtrack[t+2, path[i+1], path[i]]

        return path[::-1][:T]
