>> Write code for a barebone GPT style decoder only transformer in a new directory.

Explain the purpose of every line of code

>> Use a small toy dataset, split it into training and evaluation sets, and run both training and evaluation.

>> Add a simple visualization of the training and evaluation loss, showing that both decrease as training progresses.

>> Now refactor the code so that each major component lives in its own file. For example, keep data loading in a separate file, visualization in another, tokenization in its own file, and split model internals such as attention, MLP, positional embeddings, and layer normalization into separate modules as well. train.py and model.py can still remain, but they should call into these smaller component files.
