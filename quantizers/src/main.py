import argparse
import json
import h5py
from quantizers.FaissPQ import FaissPQ
import os
import pickle
import numpy as np

parser = argparse.ArgumentParser(prog="QuantizersH5.py", description="Quantization software that works with H5", epilog="")

parser.add_argument("param_file")
parser.add_argument("base_input_folder")
parser.add_argument("base_output_folder")

args = parser.parse_args()

params = json.load(open(args.param_file, 'r'))

BASE_INPUT_FOLDER = args.base_input_folder
BASE_OUTPUT_FOLDER = args.base_output_folder

INPUT_FILE = params.get('INPUT_FILE')
OUTPUT_FILE = params.get('OUTPUT_FILE')
TRAINING_DATASET = params.get('TRAINING_DATASET')
DOCIDS_DATASET = params.get("DOCIDS_DATASET")
TRAINING_SAMPLE_SIZE = int(params.get('TRAINING_SAMPLE_SIZE')) if params.get('TRAINING_SAMPLE_SIZE') != None else None
FULL_DATASET = params.get('FULL_DATASET')
BATCH_SIZE = int(params.get('BATCH_SIZE'))
M = int(params.get('M'))
K = int(params.get('K'))

with h5py.File(os.path.join(BASE_INPUT_FOLDER, INPUT_FILE)) as fp:

    print(f'Running {INPUT_FILE}', flush=True)

    training_dataset = fp[TRAINING_DATASET]
    full_dataset = fp[FULL_DATASET]

    vector_size = training_dataset.shape[1]

    quantizer = FaissPQ(vector_size, M, K)

    quantizer.train(training_dataset, TRAINING_SAMPLE_SIZE)

    codebook = quantizer.get_codebook()
    q_codes = quantizer.quantize(full_dataset, BATCH_SIZE)


    index_obj = {
        'codebook': codebook,
        'quantized_index': q_codes,
        'M': M,
        'K': K,
        'vector_size': vector_size,
    }

    if DOCIDS_DATASET != None:
        doc_ids = np.array(fp[DOCIDS_DATASET][:])
        index_obj['doc_ids'] = doc_ids
    
    else:
        doc_ids = [str(i) for i in range(fp[FULL_DATASET].shape[0])]
        index_obj['doc_ids'] = doc_ids

    # os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # output_path = os.path.join(OUTPUT_FILE, 'quantized_index.pickle')
    pickle.dump(index_obj, open(os.path.join(BASE_OUTPUT_FOLDER, OUTPUT_FILE), 'wb'))


