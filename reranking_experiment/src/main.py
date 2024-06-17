from pathlib import Path
import sys
from typing import Union
sys.path.append('../../')
from quantized_fast_forward.fast_forward.index import FaissPQIndex, H5Index, Mode
from quantized_fast_forward.fast_forward.encoder import QueryEncoder
from quantized_fast_forward.fast_forward.ranking import Ranking
from datasets import get_dataset
from ir_measures import calc_aggregate
from experiments import EXPERIMENTS
import pandas as pd
import os
from utils import get_size
import argparse

parser = argparse.ArgumentParser('reranking_experiment', description="Reranking experiment", epilog='Reranking experiment')

parser.add_argument('experiment_name')
parser.add_argument('-o', '--output', default=None, type=str)
parser.add_argument('-s', '--start', default=0, type=int)
parser.add_argument('-e', '--end', default=None, type=int)
args = parser.parse_args()


experiment_name = args.experiment_name
start_index = args.start
end_index = args.end
# GET THE EXPERIMENT DEPENDENCIES
EXP = EXPERIMENTS[sys.argv[1]]
# METRICS = [nDCG@10, AP(rel=1)@1000, RR(rel=1)@10]
METRICS = EXP['metrics']
ALPHA_RANGE = EXP['alpha_range']
TOP_K_RANGE = EXP.get('top_k_range', [None])
DATASET_SIZE = EXP['dataset_size']
CONFIGURATIONS = EXP['configurations']

if end_index == None:
    end_index = len(CONFIGURATIONS)


CONFIGURATIONS_SUBSET = CONFIGURATIONS[start_index : end_index]

OUTPUT_FILE = EXP['output_file'] if args.output == None else args.output

print('Loading dataset', flush=True)
queries, qrel = get_dataset(EXP['dataset'])
print('Dataset loaded', flush=True)
queries_processed = {x.query_id: x.text for x in queries}



def perform_experiment(index: Union[FaissPQIndex, H5Index]):
    """
    Function that performs one instance of the experiment.
    Keyword arguments:
        index_path -- the path to the quantized index to be used for the experiments.
    """

    if len(TOP_K_RANGE) < 1 or TOP_K_RANGE[0] == None:
        result = index.get_scores(
            EXP['ranking'](),
            queries_processed,
            alpha=ALPHA_RANGE,
        )
        result = {None: result}
    else:
        result = index.get_scores(
            EXP['ranking'](),
            queries_processed,
            alpha=ALPHA_RANGE,
            top_k=TOP_K_RANGE
        )


    return result



table_results = pd.DataFrame(columns=["Model", "M", "K", "Training Sample Size", "Training Sample Size (%)", "Top K", "Alpha", "nDCG@10", "AP@1000", "RR@10", "Dense Index Size (MB)"])

models = []
M_values = []
K_values = []
training_sample_sizes = []
training_sample_perc = []
alpha_values = []
top_k_values = []
nDCG_scores = []
AP_scores = []
RR_scores = []
dense_index_sizes = []

for i, configuration in enumerate(CONFIGURATIONS_SUBSET):

    print(f"Running {i + 1} / {len(CONFIGURATIONS_SUBSET)}", flush=True)
    
    # Run the experiment for one configuration

    index_path = configuration['index_path']
    index = configuration['get_index'](index_path)

    # handle the case of double index path
    if isinstance(index_path, tuple):
        print(index_path[0], flush=True)
        print(index_path[1], flush=True)
    else:
        print(index_path, flush=True)

    m = configuration['m']
    k = configuration['k']
    training_size = configuration['training_size']
    model_name = configuration['model_name']


    # HANDLE THE CASE WHEN TOP_K_RANGE is None

    for top_k in TOP_K_RANGE:
        for alpha in ALPHA_RANGE:
            models.append(model_name)
            M_values.append(m)
            K_values.append(k)
            training_sample_sizes.append(training_size)
            training_sample_perc.append('{:.2f}%'.format(training_size * 100 / DATASET_SIZE))
            alpha_values.append(alpha)
            top_k_values.append(top_k)

    if index == None:
        for top_k in TOP_K_RANGE:
            for alpha in ALPHA_RANGE:
                # use -1 as placeholder, if the index file does not exist
                nDCG_scores.append(-1)
                AP_scores.append(-1)
                RR_scores.append(-1)
                dense_index_sizes.append(-1)
            
    else:
        result = perform_experiment(index)

        for top_k in TOP_K_RANGE:
            for alpha in ALPHA_RANGE:
                run_results = calc_aggregate(METRICS.values(), qrel, result[top_k][alpha].run)
                nDCG_scores.append(run_results[METRICS['ndcg']])
                AP_scores.append(run_results[METRICS['ap']])
                RR_scores.append(run_results[METRICS['rr']])

                # handle the case of composed index path (for IVFPQ augmentation)

                if isinstance(index_path, tuple):
                    size0 = get_size(index_path[0], 'mb')
                    size1 = get_size(index_path[1], 'mb')
                    dense_index_sizes.append(f'{size0} | {size1}')
                else:
                    dense_index_sizes.append(get_size(index_path, 'mb'))



table_results['Model'] = models
table_results['M'] = M_values
table_results['K'] = K_values
table_results['Training Sample Size'] = training_sample_sizes
table_results['Training Sample Size (%)'] = training_sample_perc
table_results['Alpha'] = alpha_values
table_results['Top K'] = top_k_values
table_results['nDCG@10'] = nDCG_scores
table_results['AP@1000'] = AP_scores
table_results['RR@10'] = RR_scores
table_results['Dense Index Size (MB)'] = dense_index_sizes

table_results.to_csv(OUTPUT_FILE)
