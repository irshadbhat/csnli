# NMT: Neural Machine Translation/Transliteration

## Requirements

```bash
pip install -r requirements.txt
```

## Quickstart

### Step 0: Preprocess data (Only for Transliteration)

```bash
python preprocess.py input_file output_file lang  #lang options are hi (Hindi), en (English), ko (Korean) and th (Thai).
```

### Step 1: Build Vocabulary

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/dsrc2tgt
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `src2tgt.train.pt`: serialized PyTorch file containing training data
* `src2tgt.valid.pt`: serialized PyTorch file containing validation data
* `src2tgt.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
python train.py -data data/src2tgt -save_model src2tgt-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpuid 0` to use (say) GPU 0.

### Step 3: Translate/Transliterate

```bash
python file_translate.py -model src2tgt-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt
python file_transliterate.py -model src2tgt-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt
```

## Translation Parameters (Best Found - parameter search)

* -encoder_type brnn
* -layers 2
* -rnn_size 1024
* -word_vec_size 512
* -dropout 0.5
* -rnn_type LSTM
* -global_attention general (Luong)
* -optim sgd or adam
* -learning_rate 1.0 (sgd), 0.001 (adam)

## Transliteration Parameters (Best Found - parameter search)

* -encoder_type brnn
* -layers 1
* -rnn_size 512
* -word_vec_size 64
* -dropout 0.5
* -rnn_type LSTM
* -global_attention general (Luong)
* -optim sgd or adam
* -learning_rate 1.0 (sgd), 0.001 (adam)

## Pretrained Embeddings

Using pretrained embeddings is highly recommended. It can improve the translation upto 2 BLEU score.

```bash
python tools/embeddings_to_torch.py -emb_file Pretrained_embedding -output_file torch-embedding  -type word2vec -dict_file path/to/vocab/file (from Step 1)
```
Use this pretrained embedding (`torch-embedding`) while training (add -pre_word_vecs_enc torch-embedding.enc.pt-pre_word_vecs_dec torch-embedding.dec.pt to `train.py` in Step 2)

## Additional Hacks

### Tokenization and Lowercasing

Since the last layer of the NMT network has the dimensions of target vocabulary, it is very important to tokenize and lowercase the data if eiath source or target language is English or any other Roman script language. Proper tokenization and lowercasing can double the training speed and can also improve the BLEU score.

### Sequence-to-Sequence: Word Sequence to Word Sequence VS Char Sequence to Word Sequence

Training a char sequence to word sequence for Thai to English and Korean to English improvesd the BLEU score by approx. 2 as compared to training word sequence to word sequence model.This also eliminated the segmentation issue for Thai.
