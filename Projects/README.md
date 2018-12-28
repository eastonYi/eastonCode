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
