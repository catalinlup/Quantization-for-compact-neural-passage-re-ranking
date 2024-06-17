import ir_datasets
from ir_datasets.formats.base import GenericQuery
from ir_datasets.formats.trec import TrecQrel
import pandas as pd


DATASET_BASE_FOLDER = "/scratch/clupau/reranking"


def get_dataset(dataset: str):
    if not (dataset in ['trec-dl-2019-sample', 'trec-dl-2020', 'trec-dl-2019', 'dev', 'dev-local-mock', 'dev-colbert-local-mock', 'dev-local']):
        raise Exception('Invalid dataset')

    if dataset == 'trec-dl-2020':
        psg20 = ir_datasets.load('msmarco-passage/trec-dl-2020')

        queries_list = []
        qrels_list = []

        for query in psg20.queries_iter():
            queries_list.append(GenericQuery(query_id=str(query.query_id), text=query.text))

        for qrel in psg20.qrels_iter():
            qrels_list.append(TrecQrel(query_id=str(qrel.query_id), doc_id=str(qrel.doc_id), relevance=int(qrel.relevance), iteration=str(qrel.iteration)))

        return queries_list, qrels_list
    
    elif dataset == 'trec-dl-2019':
        queries = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/queries/msmarco-test2019-queries.tsv', sep='\t', header=None)
        qrels = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/qrels/msmarco_psg_trec_2019_qrel.tsv', sep='\t', header=None)

        queries_list = []
        qrels_list = []
        
        for i in range(len(queries)):
            queries_list.append(GenericQuery(str(queries.iloc[i][0]), queries.iloc[i][1]))

        for i in range(len(qrels)):
            qrels_list.append(TrecQrel(query_id=str(qrels.iloc[i][0]), doc_id=str(qrels.iloc[i][2]), relevance=int(qrels.iloc[i][3]), iteration=str(qrels.iloc[i][1])))

        return queries_list, qrels_list
    

    elif dataset == 'trec-dl-2019-sample':
        queries = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/queries/msmarco-test2019-queries.sample.small.tsv', sep='\t', header=None)
        qrels = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/qrels/msmarco_psg_trec_2019_qrel.sample.small.tsv', sep='\t', header=None)

        queries_list = []
        qrels_list = []
        
        for i in range(len(queries)):
            queries_list.append(GenericQuery(str(queries.iloc[i][0]), queries.iloc[i][1]))

        for i in range(len(qrels)):
            qrels_list.append(TrecQrel(query_id=str(qrels.iloc[i][0]), doc_id=str(qrels.iloc[i][2]), relevance=int(qrels.iloc[i][3]), iteration=str(qrels.iloc[i][1])))

        return queries_list, qrels_list


    elif dataset == 'dev':
        queries = pd.read_csv(f'{DATASET_BASE_FOLDER}/msmarco_psg_queries.dev.sample.tsv', sep='\t', header=None)
        qrels = pd.read_csv(f'{DATASET_BASE_FOLDER}/msmarco_psg_dev_qrel.sample.tsv', sep='\t', header=None)

        queries_list = []
        qrels_list = []
        
        for i in range(len(queries)):
            queries_list.append(GenericQuery(str(queries.iloc[i][0]), queries.iloc[i][1]))

        for i in range(len(qrels)):
            qrels_list.append(TrecQrel(query_id=str(qrels.iloc[i][0]), doc_id=str(qrels.iloc[i][2]), relevance=int(qrels.iloc[i][3]), iteration=str(qrels.iloc[i][1])))

        return queries_list, qrels_list
    
    elif dataset == 'dev-local':
        queries = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/reranking_dev_sample_original/msmarco_psg_queries.dev.sample.tsv', sep='\t', header=None)
        qrels = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/reranking_dev_sample_original/msmarco_psg_dev_qrel.sample.tsv', sep='\t', header=None)

        queries_list = []
        qrels_list = []
        
        for i in range(len(queries)):
            queries_list.append(GenericQuery(str(queries.iloc[i][0]), queries.iloc[i][1]))

        for i in range(len(qrels)):
            qrels_list.append(TrecQrel(query_id=str(qrels.iloc[i][0]), doc_id=str(qrels.iloc[i][2]), relevance=int(qrels.iloc[i][3]), iteration=str(qrels.iloc[i][1])))

        return queries_list, qrels_list
    
    elif dataset == 'dev-local-mock':
        queries = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/queries/msmarco_psg_queries.dev.sample.small.tsv', sep='\t', header=None)
        qrels = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/qrels/msmarco_psg_dev_qrel.sample.small.tsv', sep='\t', header=None)

        queries_list = []
        qrels_list = []
        
        for i in range(len(queries)):
            queries_list.append(GenericQuery(str(queries.iloc[i][0]), queries.iloc[i][1]))

        for i in range(len(qrels)):
            qrels_list.append(TrecQrel(query_id=str(qrels.iloc[i][0]), doc_id=str(qrels.iloc[i][2]), relevance=int(qrels.iloc[i][3]), iteration=str(qrels.iloc[i][1])))

        return queries_list, qrels_list

    elif dataset == 'dev-colbert-local-mock':
        queries = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/queries/colbert_test_0_100_query.tsv', sep='\t', header=None)
        qrels = pd.read_csv(f'/home/catalinlup/MyWorkspace/MasterThesis/datasets/qrels/colbert_test_0_100_qrel.tsv', sep='\t', header=None)

        queries_list = []
        qrels_list = []
        
        for i in range(len(queries)):
            queries_list.append(GenericQuery(str(queries.iloc[i][0]), queries.iloc[i][1]))

        for i in range(len(qrels)):
            qrels_list.append(TrecQrel(query_id=str(qrels.iloc[i][0]), doc_id=str(qrels.iloc[i][2]), relevance=int(qrels.iloc[i][3]), iteration=str(qrels.iloc[i][1])))

        return queries_list, qrels_list
