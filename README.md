# Quantization-for-compact-neural-passage-re-ranking

This repository contains the source code written to perform the necessary experiments for the thesis. The repository has the following structure:

- **experiment_notebooks** - contains a collection of jupyter notebooks that were used for data pre-processing and analysis
- **quantized_fast_forward** - it contains a modified version of the [fast-forward-indexes](https://github.com/mrjleo/fast-forward-indexes) library that supports quantized indices.
- **quantizers** - it contains the source code of the FAISS wrapper used to created the PQ indices
- **reranking_experiment** - contains the source code used in the e2e evaluation of the various two-stage retrieval setups
- **wolfpq_notebooks** - contains notebooks used to train WolfPQ as well as created WolfPQ-quantized indices

