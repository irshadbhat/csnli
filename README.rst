CSNLI
=====

----

Neural language identification and normalisation in code switching data tailored with a three-step decoding process

Requirements
^^^^^^^^^^^^

::

    pip install -r requirements.txt
    python build_viterbi.py build_ext --inplace
    
Download models from `csnli-models`_.

.. _`csnli-models`: https://bitbucket.org/irshadbhat/csnli-models/src

::

    bunzip2 lm/*
    bunzip2 dicts/*
    bunzip2 lid_models/*
    bunzip2 nmt_models/*


Three Step Decoding
^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> from three_step_decoding import *
    >>> tsd = ThreeStepDecoding('lid_models/hinglish', htrans='nmt_models/rom2hin.pt', etrans='nmt_models/eng2eng.pt')
    >>> print '\n'.join(['\t'.join(x) for x in tsd.tag_sent(u'i thght mosam dfrnt hoga bs fog h')])
    i   i   en
    thght   thought en
    mosam   मौसम hi
    dfrnt   different   en
    hoga    होगा  hi
    bs  बस  hi
    fog fog en
    h   है   hi
    >>> print '\n'.join(['\t'.join(x) for x in tsd.tag_sent(u'kafi dprsng situation hai yar')])
    kafi    काफी  hi
    dprsng  depressing  en
    situation   situation   en
    hai है   hi
    yar यार  hi


Language Identification
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> from lang_tagger import *
    >>> lid = LID(model='lid_models/hinglish', etrans='nmt_models/eng2eng.pt', htrans='nmt_models/rom2hin.pt')
    >>> lid.tag_sent(u'i thght mosam dfrnt hoga bs fog h'.split())
    [(u'i', u'en'), (u'thght', u'en'), (u'mosam', u'hi'), (u'dfrnt', u'en'), (u'hoga', u'hi'), (u'bs', u'hi'), (u'fog', u'en'), (u'h', u'hi')]
    >>> lid.tag_sent(u'kafi dprsng situation hai yar'.split())
    [(u'kafi', u'hi'), (u'dprsng', u'en'), (u'situation', u'en'), (u'hai', u'hi'), (u'yar', u'hi')]


Work with files
^^^^^^^^^^^^^^^

::

    python lang_tagger.py --test test_file --load lid_models/hinglish --etrans nmt_models/eng2eng.pt --htrans nmt_models/rom2hin.pt --out output_file

    python three_step_decoding.py --test test_file --lid lid_models/hinglish --etrans nmt_models/eng2eng.pt --htrans nmt_models/rom2hin.pt --out output_file


Train your own models
^^^^^^^^^^^^^^^^^^^^^

::

    python lang_tagger.py --help
    
    Language Identification System
    
    optional arguments:
      -h, --help            show this help message and exit
      --dynet-seed SEED
      --train TRAIN         CONLL/TNT Train file
      --dev DEV             CONLL/TNT Dev/Test file
      --test TEST           Raw Test file
      --eng-pretrained-embd EEMBD
                            Pretrained word2vec Embeddings
      --hin-pretrained-embd HEMBD
                            Pretrained word2vec Embeddings
      --elimit ELIMIT       load top-n English word vectors (default=all vectors,
                            recommended=400k)
      --hlimit HLIMIT       load top-n Hindi word vectors (default=all vectors,
                            recommended=200k)
      --trainer TRAINER     Trainer [cysgd|momsgd|adam|adadelta|adagrad|amsgrad]
      --activation-fn ACT_FN
                            Activation function [tanh|relu|sigmoid]
      --iter ITER           No. of Epochs
      --bvec BVEC           1 if binary embedding file else 0
      --etrans ETRANS       OpenNMT English Transliteration Model
      --htrans HTRANS       OpenNMT Hindi Transliteration Model
      --save-model SAVE_MODEL
                            Specify path to save model
      --load-model LOAD_MODEL
                            Load Pretrained Model
      --output-file OFILE   Output File

Cite
^^^^

Any publication reporting the work done using this data should cite the following papers:

::

    @inproceedings{bhat2017joining, 
      title={Joining Hands: Exploiting Monolingual Treebanks for Parsing of Code-mixing Data},
      author={Bhat, Irshad and Bhat, Riyaz A and Shrivastava, Manish and Sharma, Dipti},
      booktitle={Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers},
      volume={2},
      pages={324--330},
      year={2017}
    }
    
    @inproceedings{bhat20`18universal,
      title={Universal Dependency Parsing for Hindi-English Code-Switching},
      author={Bhat, Irshad and Bhat, Riyaz A and Shrivastava, Manish and Sharma, Dipti},
      booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      volume={1},
      pages={987--998},
      year={2018}
    }

Contact
^^^^^^^

::

    Irshad Ahmad Bhat
    MS-CSE IIITH, Hyderabad
    bhatirshad127@gmail.com
    irshad.bhat@research.iiit.ac.in

