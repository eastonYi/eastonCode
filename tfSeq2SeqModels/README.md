# Seq2seq Model

the seq_label is right padded with `<eos>`.
the decoder output is seq_label, and input is seq_label left padded with `<sos>`.
output: 1,2,3,4,\<eos>
input: \<sos>, 1,2,3,4,\<eos>
