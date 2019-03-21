This is the demo project folder which uses the python liberaries in `eastonCode`.
A project contains the `main.py` which is used to run the programmes.

For example, in the `aishell` dir, you can run the training and infer with the blow commands:
```bash
python ../../main.py -c configs/xx.yaml
python ../../main.py -m infer --gpu 0 -c configs/xx.yaml
```

## model parameters load from dirfferent checkpoints
Take the `infer.py` as an example where you need to load two different models from different checkpoints: the CTC model from
> Projects/asr-tf/exp/aishell/models/ctc_lm_rna4.yaml/checkpoint

and language model from
> Projects/nmt-tf/exp/aishell2/models/aishell2.yaml/checkpoint

These two models are trained seperately and need to use beam search with language model.
We first use a subargs: `args.args_lm` to create a language model and a saver with its variables:
```python
if args.dirs.lm_config:
    from utils.configReader import AttrDict
    import yaml
    args.args_lm = AttrDict(yaml.load(open(args.dirs.lm_config)))
    args.args_lm.dim_output = len(args.token2idx)
    args.args_lm.list_gpus = args.list_gpus
    from tfSeq2SeqModels.languageModel import LanguageModel
    args.Model_LM = LanguageModel

model_lm = args.Model_LM(
    tensor_global_step,
    is_train=False,
    args=args.args_lm)

args.lm = model_lm
saver_lm = tf.train.Saver(model_lm.variables())
```
We name the language model with `args.lm` and it can be easily accessed by when creating the ctc model with `args`.

Then, we start to build the ctc model and a saver with its variabels:
```python
model_infer = args.Model(
    tensor_global_step,
    encoder=args.model.encoder.type,
    decoder=args.model.decoder.type,
    decoder2=args.model.decoder2.type,
    is_train=False,
    args=args)
saver = tf.train.Saver(model_infer.variables())
```
Within the construction, we could use the object's functions:
```python
# within building the ctc model
class CTCModel:
    def __init__():
        ...
        if args.model.shallow_fusion:
            logging.info('load language model object: {}'.format(args.lm_obj))
            self.lm = args.lm_obj
        ...
```

# Models
<a href="/README.html" target="_blank">
  <img class="aligncenter" alt="framework classes" src="images/2018/10/framework-classes.png" width="700" height="300" />
</a>
## LSTM_Model
It is the father of Seq2Seq models. It has two class args: `num_Instances` and `num_Model`.
`num_Instances` is used to count how many times the `build_model()` has been called;
`num_Model` is used to cound how many times the `build_single_model()` has been called
Usually, the `build_model()` will be called twice for building trainmodel and infer model.
Meanwhile, the `build_single_model()` will be called as many times as the GPU numer within the  `build_model()`.

## Seq2Seq Models
it has the encoder folder and decoder folder

### embedding
We explicitly create encoder embedding and decoder embedding in the seq2seq framwork(not in encoder or decoder).
- using outside ebedding
When build seq2seq models, we could transfer the outer embeddings into the construction.
- create embedding in seq2seq
Consider that not all encoders and  decoders need the embedding, such as ASR only need decoder embedding and CTC need neither embedding. So we control whether to create it by `en(de)coder.size_embedding`.

## CTC Models
it uses the encoder folder.
The most recommand model to be a baseline for its simplicity, high efficient and performance.

## RNA Model
it has the encoder-decoder framwork but use the ctc loss.

## Encoders
- conv_lstm_lh
if using blstm, the num_cell_units represent the sum of the two directions cell numbers.

## Decoders
deocder has the embedding function
```python
def embedding(self, ids):
    if self.embed_table:
        embeded = tf.nn.embedding_lookup(self.embed_table, ids)
    else:
        embeded = tf.one_hot(ids, self.args.dim_output, dtype=tf.float32)

    return embeded
```

# vocab
```
<pad>
<unk>
<sos>
<eos>
.
.
.
<blk>
```
`<sos>`, `<eos>` is for Transformer, Seq2seq.
`<blk>` is for RNA, CTC and CTC-LM

# Tools
- Data Tools
the dataset and dataloader class. They are iterable objects and return data in numpy forms.

- Config Tools
read the .yaml files
the `arguments.py` load the general configs into an object `args`. We process some parameters here.
