# CTC-LM experiments
This repo is for the code and experiments of my paper CTC-LM ![link].



## phone as acoustic label
- generating phone label according to the trans
```python
from pypinyin import pinyin

with open('text', encoding='utf-8') as f, open('train_pinyin.csv', 'w', encoding='utf-8') as fw:
    for _, line in zip(range(99999999), f):
        uttid, trans = line.strip().split(',')
        trans = trans.replace(' ', '')
        trans = ' '.join(x[0] for x in pinyin(trans, style=Style.TONE3))
        line = uttid + ',' + trans
        fw.write(line+ '\n')
```

- generating phone vocab
```bash
python eastonCode/utils/vocab.py --input train_pinyin.csv --output vocab.txt
head -2000 vocab.txt > vocab_2k.txt
```
select the top 2k of the vocab.txt

- training the CTC-LM with double target sequences

## pipline training
- training the encoder with CTC
- training the CTC-LM with fixed encoder.
