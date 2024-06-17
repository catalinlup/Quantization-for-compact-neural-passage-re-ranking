from rankings import get_msmarco_dev_sample_bm25_ranking, get_msmarco_dev_sample_bm25_ranking_local, get_msmarco_dev_colbert_sample_bm25_ranking_local, get_trec_2019_bm25_ranking_local, get_trec_2020_bm25_ranking_local, get_trec_2019_bm25_ranking_local_sample
from encoders import TCT_COLBERT_ENCODER, AGG_RETRIEVER_ENCODER, COLBERT_ENCODER
from pathlib import Path
from quantized_fast_forward.fast_forward.index import FaissPQIndex, H5Index, FaissIVFPQIndex, ColBertH5Index, WolfPQIndex
import os
from ir_measures import nDCG, AP, RR
import numpy as np


BASE_FOLDER = "/scratch/clupau"
#BASE_FOLDER = "/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices"

# K-M GridSearch

EXPERIMENTS = {
    'k_m_gridsearch': {
        'alpha_range': [0],
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
        'dataset': 'dev',
        'dataset_size': 8800000,
        'configurations': [],
        'output_file': '../results/k_m_gridsearch.csv'
    }
}


for training_size in {200000}:
    for m in {8, 16, 24, 32, 48, 64, 96}:
        for k in {256, 512, 1024, 2048, 4096}:
            index_path = f'{BASE_FOLDER}/agg_m_{m}_k_{k}_{training_size}.pickle'
            
           
            EXPERIMENTS['k_m_gridsearch']['configurations'].append({'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path),AGG_RETRIEVER_ENCODER) if os.path.exists(index_path) else None, 'index_path': index_path, 'm': m, 'k': k, 'training_size': training_size, 'model_name': 'BM25 + AggretrieverPQ'})


for training_size in {200000}:
    for m in {8, 16, 24, 32, 48, 64, 96}:
        for k in {256, 512, 1024, 2048, 4096}:
            index_path = f'{BASE_FOLDER}/tct_m_{m}_k_{k}_{training_size}.pickle'

            EXPERIMENTS['k_m_gridsearch']['configurations'].append({'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER) if os.path.exists(index_path) else None, 'index_path': index_path, 'm': m, 'k': k, 'training_size': training_size, 'model_name': 'BM25 + TctColBertPQ'})


# K-M GridSearch Leftovers
            
EXPERIMENTS['k_m_gridsearch_leftovers'] = {
        'alpha_range': [0],
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
        'dataset': 'dev',
        'dataset_size': 8800000,
        'configurations': [],
        'output_file': '../results/k_m_gridsearch_leftovers.csv'
    }

for training_size in {200000}:
    for m in {2, 4}:
        for k in {256, 512, 1024, 2048, 4096}:
            index_path = f'{BASE_FOLDER}/agg_m_{m}_k_{k}_{training_size}.pickle'
            
           
            EXPERIMENTS['k_m_gridsearch_leftovers']['configurations'].append({'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path),AGG_RETRIEVER_ENCODER) if os.path.exists(index_path) else None, 'index_path': index_path, 'm': m, 'k': k, 'training_size': training_size, 'model_name': 'BM25 + AggretrieverPQ'})


for training_size in {200000}:
    for m in {2, 4}:
        for k in {256, 512, 1024, 2048, 4096}:
            index_path = f'{BASE_FOLDER}/tct_m_{m}_k_{k}_{training_size}.pickle'

            EXPERIMENTS['k_m_gridsearch_leftovers']['configurations'].append({'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER) if os.path.exists(index_path) else None, 'index_path': index_path, 'm': m, 'k': k, 'training_size': training_size, 'model_name': 'BM25 + TctColBertPQ'})


# MOCK EXPERIMENT

EXPERIMENTS['mock'] = {
    'alpha_range': [0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
    'dataset': 'dev',
    'dataset_size': 8800000,
    'output_file': '../results/mock2.csv',
    'configurations': [
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'{BASE_FOLDER}/agg_m_6_k_256_10000.pickle',
            'm': 6,
            'k': 256,
            'training_size': 10000,
            'model_name': 'BM25 + AggretrieverPQ'
        },
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'{BASE_FOLDER}/tct_m_6_k_256_10000.pickle',
            'm': 6,
            'k': 256,
            'training_size': 10000,
            'model_name': 'BM25 + TctColBertPQ'
        }
    ]
}

# Baseline Experiment

EXPERIMENTS['baseline'] = {
    'alpha_range': [0.0, 0.1, 0.3, 1.0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
    'dataset': 'dev',
    'dataset_size': 8800000,
    'output_file': '../results/baseline.csv',
    'configurations': [
        {
            'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'{BASE_FOLDER}/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
            'm': -1,
            'k': -1,
            'training_size': 0,
            'model_name': 'BM25 + Aggretriever'
        }
    ]
}

# Alpha Optimization Experiment Baseline

EXPERIMENTS['alpha_optimization_baseline'] = {
        'alpha_range': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
        'dataset': 'dev',
        'dataset_size': 8800000,
        'configurations': [
            {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'{BASE_FOLDER}/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + Aggretriever'
            },
            {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'{BASE_FOLDER}/faiss.msmarco-v1-passage.tct_colbert.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + TctColBert'
            }
        ],
        'output_file': '/scratch/clupau/alpha/alpha_optimization_baseline.csv'
}



# Alpha Optimization Experiment

EXPERIMENTS['alpha_optimization'] = {
        'alpha_range': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
        'dataset': 'dev',
        'dataset_size': 8800000,
        'configurations': [
            {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'{BASE_FOLDER}/agg_m_96_k_4096_200000.pickle',
            'm': 96,
            'k': 4096,
            'training_size': 200000,
            'model_name': 'BM25 + AggretrieverPQ'
        },
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'{BASE_FOLDER}/tct_m_96_k_4096_200000.pickle',
            'm': 96,
            'k': 4096,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        }
        ],
        'output_file': '/scratch/clupau/alpha/alpha_optimization.csv'
}


# Alpha Optimization Tct Fine Grained Experiment

EXPERIMENTS['alpha_optimization_tct'] = {
        'alpha_range': list(np.linspace(0, 0.2, 20)),
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
        'dataset': 'dev',
        'dataset_size': 8800000,
        'configurations': [
            {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'{BASE_FOLDER}/tct_m_96_k_4096_200000.pickle',
                'm': 96,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
            },
            {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'{BASE_FOLDER}/faiss.msmarco-v1-passage.tct_colbert.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + TctColBert'
            }
        ],
        'output_file': '/scratch/clupau/alpha/alpha_optimization_tct.csv'
}


# Alpha Optimization Agg Fine Grained Experiment

EXPERIMENTS['alpha_optimization_agg'] = {
        'alpha_range': list(np.linspace(0, 0.5, 30)),
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
        'dataset': 'dev',
        'dataset_size': 8800000,
        'configurations': [
            {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'{BASE_FOLDER}/agg_m_96_k_4096_200000.pickle',
            'm': 96,
            'k': 4096,
            'training_size': 200000,
            'model_name': 'BM25 + AggretrieverPQ'
        },
        {
            'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'{BASE_FOLDER}/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
            'm': -1,
            'k': -1,
            'training_size': 0,
            'model_name': 'BM25 + Aggretriever'
        },
        ],
        'output_file': '/scratch/clupau/alpha/alpha_optimization_agg.csv'
}



# K-M Grid Search With Interpolation

EXPERIMENTS['k_m_gridsearch_interpolation'] = {
        'alpha_range': [0.1, 0.3], # change to optimal ALPHA
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
        'dataset': 'dev',
        'dataset_size': 8800000,
        'configurations': [],
        'output_file': '/scratch/clupau/gs_results_interpolation/k_m_gridsearch_interpolation.csv'
}

for training_size in {200000}:
    for m in {8, 16, 24, 32, 48, 64, 96}:
        for k in {256, 512, 1024, 2048, 4096}:
            index_path = f'{BASE_FOLDER}/agg_m_{m}_k_{k}_{training_size}.pickle'
            
           
            EXPERIMENTS['k_m_gridsearch_interpolation']['configurations'].append({'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path),AGG_RETRIEVER_ENCODER) if os.path.exists(index_path) else None, 'index_path': index_path, 'm': m, 'k': k, 'training_size': training_size, 'model_name': 'BM25 + AggretrieverPQ'})


for training_size in {200000}:
    for m in {8, 16, 24, 32, 48, 64, 96}:
        for k in {256, 512, 1024, 2048, 4096}:
            index_path = f'{BASE_FOLDER}/tct_m_{m}_k_{k}_{training_size}.pickle'

            EXPERIMENTS['k_m_gridsearch_interpolation']['configurations'].append({'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER) if os.path.exists(index_path) else None, 'index_path': index_path, 'm': m, 'k': k, 'training_size': training_size, 'model_name': 'BM25 + TctColBertPQ'})


# IVFPQ MOCK
EXPERIMENTS['ivfpq_mock'] = {
    'alpha_range': [0.0],
    'top_k_range': [0, 30000],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local_sample(),
    'dataset': 'trec-dl-2019-sample',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: FaissIVFPQIndex.from_disk(Path(index_path[0]), Path(index_path[1]) ,TCT_COLBERT_ENCODER),
            'index_path': ('/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5','/home/catalinlup/MyWorkspace/MasterThesis/datasets/ivf_quantized_indices/test_run_tct.index'),
            'm': 8,
            'k': 256,
            'training_size': 39000,
            'model_name': 'BM25 + TctColBert + IVFPQ Aug'
        }
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/ivfpq_mock_test.csv'
}

# ColBERT Mock

EXPERIMENTS['colbert_mock'] = {
    'alpha_range': [0.0, 1.0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_colbert_sample_bm25_ranking_local(),
    'dataset': 'dev-colbert-local-mock',
    'dataset_size': 8800000,
    'configurations': [
         {
                'get_index': lambda index_path: ColBertH5Index.from_disk(Path(index_path), COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/index_0_100.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + ColBert'
        },
        {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + TctColBert'
        }
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/colbert_mock.csv'
}


EXPERIMENTS['colbert'] = {
    'alpha_range': [0.0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking(),
    'dataset': 'dev',
    'dataset_size': 8800000,
    'configurations': [
         {
                'get_index': lambda index_path: ColBertH5Index.from_disk(Path(index_path), COLBERT_ENCODER),
                'index_path': f'/scratch/clupau/colbert_h5_index/msmarco_colbert.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + ColBert'
        }
    ],
    'output_file': '/scratch/clupau/colbert_results/plain_colbert.csv'
}


# TREC-DL-2019
EXPERIMENTS['trecl-dl-2019-local'] = {
    'alpha_range': [0.0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
        # {
        #     'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_96_k_4096_200000.pickle',
        #     'm': 96,
        #     'k': 4096,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + AggretrieverPQ'
        # },
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_4096_200000.pickle',
            'm': 96,
            'k': 4096,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        }

        # {
        #     'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
        #     'm': -1,
        #     'k': -1,
        #     'training_size': 0,
        #     'model_name': 'BM25 + Aggretriever'
        # },
        # {
        #     'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5',
        #     'm': -1,
        #     'k': -1,
        #     'training_size': 0,
        #     'model_name': 'BM25 + TctColBert'
        # }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/trec_dl_2019_local_3.csv'
}

# TREC-DL-2020
EXPERIMENTS['trecl-dl-2020-local'] = {
    'alpha_range': [0.0, 1.0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_96_k_4096_200000.pickle',
            'm': 96,
            'k': 4096,
            'training_size': 200000,
            'model_name': 'BM25 + AggretrieverPQ'
        },
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
            'm': 96,
            'k': 2048,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        },
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_4096_200000.pickle',
            'm': 96,
            'k': 4096,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        },
        {
            'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
            'm': -1,
            'k': -1,
            'training_size': 0,
            'model_name': 'BM25 + Aggretriever'
        },
        {
            'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5',
            'm': -1,
            'k': -1,
            'training_size': 0,
            'model_name': 'BM25 + TctColBert'
        }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/trec_dl_2020_local.csv'
}

# INTERPOLATION NO QUANTIZATION LOCAL

EXPERIMENTS['interp_no_quant'] = {
    'alpha_range': [0.1, 0.3],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
            'm': -1,
            'k': -1,
            'training_size': 0,
            'model_name': 'BM25 + Aggretriever'
        },
        {
            'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5',
            'm': -1,
            'k': -1,
            'training_size': 0,
            'model_name': 'BM25 + TctColBert'
        }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/dev_local_experiment.csv'
}

# WOLFPQ EXPERIMENTS

EXPERIMENTS['wolfpq_pre_64_1024_tct_local'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_64_k_1024_200000.pickle',
            'm': 64,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        },

          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_pre_wolfpq_64_1024_200000.pickle',
            'm': 64,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQPre' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/result_tct_64_1024_pre.csv'
}

EXPERIMENTS['wolfpq_pre_24_1024_tct_local'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_24_k_1024_200000.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        },

          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_pre_wolfpq_24_1024_200000.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQPre' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/result_tct_24_1024_pre.csv'
}

# tct_wolfpq_16_256_lip_05_alpha_00_200000.pickle

EXPERIMENTS['wolfpq_pre_16_256_tct_local'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_16_k_256_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        },

          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_pre_wolfpq_16_256_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQPre' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/result_tct_16_256_pre.csv'
}


EXPERIMENTS['tct_wolfpq_16_256_lip_05_alpha_00'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_05_alpha_00_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQ (LIP 05)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_05_alpha_00.csv'
}

# tct_wolfpq_16_256_lip_00_alpha_00_200000.pickle

EXPERIMENTS['tct_wolfpq_16_256_lip_10_alpha_00'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_10_alpha_00_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQ (LIP 10)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_10_alpha_00.csv'
}

EXPERIMENTS['tct_wolfpq_16_256_lip_00_alpha_00'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_00_alpha_00_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQ (LIP 00)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_00_alpha_00.csv'
}

EXPERIMENTS['tct_wolfpq_16_256_lip_03_alpha_00'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_03_alpha_00_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQ (LIP 03)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_03_alpha_00.csv'
}

# tct_wolfpq_16_256_lip_001_alpha_01_200000


EXPERIMENTS['tct_wolfpq_16_256_lip_001_alpha_01_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_001_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 001)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_001_alpha_01.csv'
}

EXPERIMENTS['tct_wolfpq_16_256_lip_00_alpha_01_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_00_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 00)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_00_alpha_01.csv'
}

EXPERIMENTS['tct_wolfpq_16_256_lip_005_alpha_01_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_005_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 005)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_005_alpha_01.csv'
}


EXPERIMENTS['tct_wolfpq_16_256_swa1_alpha_01_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_swa1_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSWA1'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_swa1_alpha_01_200000.csv'
}


EXPERIMENTS['tct_wolfpq_semantic_sparse_no_sparse'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_lip_00_alpha_01_sem_mdf30_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
          },
           {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_swa2_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30Sparse'
          }

        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_sem_sparse_no_sparse.csv'
}

EXPERIMENTS['tct_wolfpq_semantic_sparse_30e_hard_soft_sampling'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_swa2(30e)_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30Sparse 30 epochs'
          },
           {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_swa2(30e,argmax)_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30Sparse 30 epochs argmax'
          }

        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_semantic_sparse_30e_hard_soft_sampling.csv'
}


EXPERIMENTS['tct_16_256_lip_00_alpha_01_sem_mdf30_argmax_epoch_25'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_lip_00_alpha_01_sem_mdf30_argmax_epoch_25.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30 argmax'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_16_256_lip_00_alpha_01_sem_mdf30_argmax_epoch_25.csv'
}

EXPERIMENTS['tct_wolfpq_16_256_lip_00_alpha_01_200000_argmax'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_00_alpha_01_200000_argmax.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30 argmax'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_wolfpq_16_256_lip_00_alpha_01_200000_argmax.csv'
}



EXPERIMENTS['tct_16_256_sem_mdf30_200000'] = {
    'alpha_range': [0.05, 0.075, 0.08, 0.09, 0.1, 0.11, 0.125, 0.13, 0.135, 0.14, 0.15, 0.17, 0.2],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sem_mdf30_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_16_256_sem_mdf30_200000.csv'
}



EXPERIMENTS['tct_16_256_sw_sem_mdf30_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sw_sem_mdf30_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSWSemMDF30'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_16_256_sw_sem_mdf30_200000.csv'
}

EXPERIMENTS['tct_16_256_sw_sem_mdf25_55_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sem_mdf25_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF25'
          },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sem_mdf55_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF55'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_16_256_sw_sem_mdf25_55_200000.csv'
}

EXPERIMENTS['tct_16_256_sem_mdf5_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sem_mdf5_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF5'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_16_256_sem_mdf5_200000.csv'
}


EXPERIMENTS['tct_16_256_sem_mdf35_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sem_mdf35_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF35'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_16_256_sem_mdf35_200000.csv'
}


EXPERIMENTS['tct_24_1024_sem_mdf30_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_24_1024_sem_mdf30_200000.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_24_1024_sem_mdf30_200000.csv'
}


EXPERIMENTS['tct_pre_wolfpq_96_256_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_pre_wolfpq_96_256_200000.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQPre'
          },
          {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
                'm': 96,
                'k': 256,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
            },
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_pre_wolfpq_96_256_200000.csv'
}



# EXPERIMENTS TEST SET 2019
EXPERIMENTS['tct_wolfpq_sem_test_2019'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
        #  {
        #     'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
        #     'm': 96,
        #     'k': 2048,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + TctColBertPQ'
        # },
        #  {
        #     'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_16_k_256_200000.pickle',
        #     'm': 16,
        #     'k': 256,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + TctColBertPQ'
        # },
        # {
        #     'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sem_mdf30_200000.pickle',
        #     'm': 16,
        #     'k': 256,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
        #   },
        #    {
        #         'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #         'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
        #         'm': 96,
        #         'k': 256,
        #         'training_size': 200000,
        #         'model_name': 'BM25 + TctColBertPQ'
        #     },
        #        {
        #         'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #         'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
        #         'm': 96,
        #         'k': 1024,
        #         'training_size': 200000,
        #         'model_name': 'BM25 + TctColBertPQ'
        #     },
        #     {
        #         'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #         'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
        #         'm': 96,
        #         'k': 2048,
        #         'training_size': 200000,
        #         'model_name': 'BM25 + TctColBertPQ'
        #     },
        #   {
        #         'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #         'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_64_k_1024_200000.pickle',
        #         'm': 64,
        #         'k': 1024,
        #         'training_size': 200000,
        #         'model_name': 'BM25 + TctColBertPQ'
        #     },
        #     {
        #         'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #         'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_24_k_1024_200000.pickle',
        #         'm': 24,
        #         'k': 1024,
        #         'training_size': 200000,
        #         'model_name': 'BM25 + TctColBertPQ'
        #     },
        #     {
        #     'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_24_1024_sem_mdf30_200000.pickle',
        #     'm': 24,
        #     'k': 1024,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
        #   },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_96_256_sem_mdf30_200000.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
          }
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments_test_set_2019/tct/tct_wolfpq_sem_test_2019_v3.csv'
}


EXPERIMENTS['tct_wolfpq_sem_test_2020'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
        #  {
        #     'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
        #     'm': 96,
        #     'k': 2048,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + TctColBertPQ'
        # },
        #  {
        #     'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_16_k_256_200000.pickle',
        #     'm': 16,
        #     'k': 256,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + TctColBertPQ'
        # },
        # {
        #     'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
        #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_16_256_sem_mdf30_200000.pickle',
        #     'm': 16,
        #     'k': 256,
        #     'training_size': 200000,
        #     'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
        #   },
            # {
            #     'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_24_k_1024_200000.pickle',
            #     'm': 24,
            #     'k': 1024,
            #     'training_size': 200000,
            #     'model_name': 'BM25 + TctColBertPQ'
            # },
            # {
            #     'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_64_k_1024_200000.pickle',
            #     'm': 64,
            #     'k': 1024,
            #     'training_size': 200000,
            #     'model_name': 'BM25 + TctColBertPQ'
            # },
            # {
            #     'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            #     'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_24_1024_sem_mdf30_200000.pickle',
            #     'm': 24,
            #     'k': 1024,
            #     'training_size': 200000,
            #     'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
            # },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_pre_wolfpq_96_256_200000.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQPre'
          },
         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_96_256_sem_mdf30_200000.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemMDF30'
          }
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments_test_set_2020/tct/tct_wolfpq_sem_test_2020_v3.csv'
}




# FINAL EXPERIMENTS FULL DATASET

EXPERIMENTS['tct_pre_wolfpq_96_256_200000'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_pre_wolfpq_96_256_200000.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQPre'
          },
          {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
                'm': 96,
                'k': 256,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
            },
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/tct_pre_wolfpq_96_256_200000.csv'
}



# EXPERIMENTS TEST SET 2019
EXPERIMENTS['tct_wolfpq_final_test_2019'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_sem_mdf30_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFull'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_sem_mdf30_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFull'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_sem_mdf30_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFull'
        },
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_full_dataset_final_experiments/tct_final_correct.csv'
}

EXPERIMENTS['tct_wolfpq_final_test_2020'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_sem_mdf30_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFull'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_sem_mdf30_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFull'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_sem_mdf30_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFull'
        },
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_full_dataset_final_experiments/tct_final_correct_2020.csv'
}

EXPERIMENTS['tct_wolfpq_final_test_2019_no_sem'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_mdf30_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFull'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_mdf30_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFull'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_mdf30_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFull'
        },
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_full_dataset_final_experiments/tct_final_correct_no_sem.csv'
}

EXPERIMENTS['tct_wolfpq_final_test_2020_no_sem'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_mdf30_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFull'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_mdf30_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFull'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_mdf30_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFull'
        },
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_full_dataset_final_experiments/tct_final_correct_no_sem_2020.csv'
}

EXPERIMENTS['tct_pqi_2020_test'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_256_200000.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertPQ'
        },
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_full_dataset_final_experiments/tct_pqi_2020.csv'
}

# EXTRA PQ EXPERIMENTS

EXPERIMENTS['k_m_gridsearch_extra'] = {
        'alpha_range': [0, 0.1, 0.3],
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
        'dataset': 'dev-local',
        'dataset_size': 8800000,
        'configurations': [
            {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_48_k_4096_200000.pickle',
                'm': 48,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
            },
            {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_48_k_4096_200000.pickle',
                'm': 48,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + AggretrieverPQ'
            },
            {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_48_k_512_200000.pickle',
                'm': 48,
                'k': 512,
                'training_size': 200000,
                'model_name': 'BM25 + AggretrieverPQ'
            }

        ],
        'output_file': '../results/k_m_gridsearch_extra.csv'
}


EXPERIMENTS['extra_lip_experiments'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_07_alpha_00_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQ (LIP 07)' 
          },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_01_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 01)' 
          },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_03_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 03)' 
          },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_05_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 05)' 
          },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_07_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 07)' 
          },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_10_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 10)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/extra_lip_experiments.csv'
}

EXPERIMENTS['experiment_lip_07'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
         
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_07_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 07)' 
          },
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_00_alpha_01_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 00)' 
          }
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/extra_lip_experiments_testset.csv'
}


EXPERIMENTS['extra_lip_experiment_hard'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
    'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
    'dataset': 'dev-local',
    'dataset_size': 8800000,
    'configurations': [
       
          {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_wolfpq_16_256_lip_07_alpha_01_hard_200000.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQI (LIP 07 HARD)' 
          },
        ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments/tct/extra_lip_hard.csv'
}


EXPERIMENTS['tct_wolfpq_final_test_2019_listwise'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_mdf30_listwise_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFullListwise'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_mdf30_listwise_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFullListwise'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_mdf30_listwise_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFullListwise'
        },

        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_sem_mdf30_listwise_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFullListwise'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_sem_mdf30_listwise_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFullListwise'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_sem_mdf30_listwise_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFullListwise'
        },
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments_test_set_19_20_listwise/results_2019.csv'
}


EXPERIMENTS['tct_wolfpq_final_test_2020_listwise'] = {
    'alpha_range': [0.0, 0.1],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_mdf30_listwise_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFullListwise'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_mdf30_listwise_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFullListwise'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_mdf30_listwise_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQFullListwise'
        },

        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_96_256_sem_mdf30_listwise_epoch_1.pickle',
            'm': 96,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFullListwise'
        },

         {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_16_256_sem_mdf30_listwise_epoch_1.pickle',
            'm': 16,
            'k': 256,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFullListwise'
        },
        {
            'get_index': lambda index_path: WolfPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
            'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/wolfpq_final_tct_24_1024_sem_mdf30_listwise_epoch_1.pickle',
            'm': 24,
            'k': 1024,
            'training_size': 200000,
            'model_name': 'BM25 + TctColBertWolfPQSemFullListwise'
        },
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/wolf_pq_final_experiments_test_set_19_20_listwise/results_2020.csv'
}

EXPERIMENTS['final_baselines_2019'] = {
    'alpha_range': [0.0, 0.1, 1.0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
          {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + TctColBert'
        }
   
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/final_baselines/final_baselines_2019.csv'
}

EXPERIMENTS['final_baselines_2020'] = {
    'alpha_range': [0.0, 0.1, 1.0],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
          {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.tct_colbert.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + TctColBert'
        }
   
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/final_baselines/final_baselines_2020.csv'
}

EXPERIMENTS['k_m_gridsearch_extras'] = {
        'alpha_range': [0],
        'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=1)@1000, 'rr': RR(rel=1)@10},
        'ranking': lambda: get_msmarco_dev_sample_bm25_ranking_local(),
        'dataset': 'dev-local',
        'dataset_size': 8800000,
        'configurations': [
            {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_48_k_4096_200000.pickle',
                'm': 48,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
            },
            {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_48_k_4096_200000.pickle',
                'm': 48,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + AggretrieverPQ'
            },
            {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_48_k_512_200000.pickle',
                'm': 48,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + AggretrieverPQ'
            },
        ],
        'output_file': '../results/k_m_gridsearch_missing_points.csv'
}


EXPERIMENTS['rq2_final_experiments_2019'] = {
    'alpha_range': [0.0, 0.1, 0.3],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2019_bm25_ranking_local(),
    'dataset': 'trec-dl-2019',
    'dataset_size': 8800000,
    'configurations': [
          {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + Aggretriever'
         },
         {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_96_k_4096_200000.pickle',
                'm': 96,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + AggretrieverPQ'
         },
         {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_4096_200000.pickle',
                'm': 96,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
         },
         {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
                'm': 96,
                'k': 2048,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
         },
   
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/rq2_final_experiments/rq2_res_2019.csv'
}

EXPERIMENTS['rq2_final_experiments_2020'] = {
    'alpha_range': [0.0, 0.1, 0.3],
    'metrics': {'ndcg': nDCG@10, 'ap': AP(rel=2)@1000, 'rr': RR(rel=2)@10},
    'ranking': lambda: get_trec_2020_bm25_ranking_local(),
    'dataset': 'trec-dl-2020',
    'dataset_size': 8800000,
    'configurations': [
          {
                'get_index': lambda index_path: H5Index.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/h5_indices/faiss.msmarco-v1-passage.aggretriever-cocondenser.h5',
                'm': -1,
                'k': -1,
                'training_size': 0,
                'model_name': 'BM25 + Aggretriever'
         },
         {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), AGG_RETRIEVER_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/agg_m_96_k_4096_200000.pickle',
                'm': 96,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + AggretrieverPQ'
         },
         {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_4096_200000.pickle',
                'm': 96,
                'k': 4096,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
         },
         {
                'get_index': lambda index_path: FaissPQIndex.from_disk(Path(index_path), TCT_COLBERT_ENCODER),
                'index_path': f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/quantized_indices/tct_m_96_k_2048_200000.pickle',
                'm': 96,
                'k': 2048,
                'training_size': 200000,
                'model_name': 'BM25 + TctColBertPQ'
         },
   
    ],
    'output_file': '/home/catalinlup/MyWorkspace/MasterThesis/ExperimentsCodebase/reranking_experiment/results/rq2_final_experiments/rq2_res_2020.csv'
}