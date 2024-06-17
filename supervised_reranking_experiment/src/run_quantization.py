import torch
import h5py
# from models.WolfPQ import WolfPQ
# from models.WolfPQInit import WolfPQInit
# from models.WolfPQRotate import WolfPQRotate
# from models.WolfPQVAE import WolfPQVAE
# from models.WolfPQVAEMinDist import WolfPQVAEMinDist
# from models.WolfPQFinal import WolfPQMinDist
# from models.WolfPQEndRot import WolfPQMinDist
from models.WolfPQ import WolfPQ
import time
import pickle
import numpy as np


MODEL_PATH = '../trained_models/wolfpq_final_tct_96_256_sem_mdf30_epoch_1.pt'
DATASET_PATH = '/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5'
OUTPUT_PATH = '/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_sem_mdf30_epoch_1.pickle'
M=96
K=256
BATCH_SIZE=2000




with h5py.File(DATASET_PATH, 'r') as dataset:

    # load the dataset
    vectors = dataset['vectors']
    dim = vectors.shape[1]

    # prepare the model
    model = WolfPQ(dim, M, K, 30.0, semantic_sampling=False, pq_index_path=None)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # COMPUTE THE Q_CODES
    with torch.no_grad():
        
        vector_count = int(vectors.shape[0])
        num_batches = vector_count // BATCH_SIZE + (1 if vector_count % BATCH_SIZE > 0 else 0)

        code_batches = []

        for bi in range(0, num_batches):
            print(f'{bi + 1} / {num_batches}', flush=True)

            start_ms = time.time()

            index_start = bi * BATCH_SIZE
            index_end = min((bi + 1) * BATCH_SIZE, vector_count)

            vector_batch = np.array(vectors[index_start:index_end, :])

            vector_batch_torch = torch.from_numpy(vector_batch)
            _, code_batch_torch = model(vector_batch_torch)
            code_batch = torch.argmax(code_batch_torch, dim=-1).numpy()
            


            code_batches.append(code_batch)

            end_ms = time.time()

            print(f'Elleapsed time(s) {end_ms - start_ms}', flush=True)

        
        q_codes = np.concatenate(code_batches, axis=0)
        


        # GET THE CODEBOOK
        codebook = (model.codebook @ model.rotation_matrix).numpy()


        # BUILD THE INDEX OBJECT

        index_obj = {
            'codebook': codebook,
            'quantized_index': q_codes,
            'M': M,
            'K': K,
            'vector_size': dim,
            'doc_ids': np.array(dataset['docids'][:])
        }


        # SAVE THE INDEX OBJECT
        pickle.dump(index_obj, open(OUTPUT_PATH, 'wb'))


