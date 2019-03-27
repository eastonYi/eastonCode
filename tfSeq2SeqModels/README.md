# Seq2seq Model

the seq_label is right padded with `<eos>`.
the decoder output is seq_label, and input is seq_label left padded with `<sos>`.
output: 1,2,3,4,\<eos>
input: \<sos>, 1,2,3,4,\<eos>


# Two Type Model
Basicly, there are two kind of models: 1) models used label with `<eos>`, e.g. Transformer and 2) models do not need `<eos>`, e.g. CTC.
Two different with different vocab and the data. Further, they cannot share the same language model.
It is not such convenient.
