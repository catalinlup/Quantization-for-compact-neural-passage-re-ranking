from pathlib import Path
import sys
sys.path.append('../../')
from quantized_fast_forward.fast_forward.ranking import Ranking


RANKING_BASE_FOLDER = "/scratch/clupau/reranking"


def get_trec_2020_bm25_ranking():
    return Ranking.from_file(Path(f"{RANKING_BASE_FOLDER}/msmarco_psg_bm25_rankings.trec_2020.tsv"))

def get_trec_2019_bm25_ranking():
    return Ranking.from_file(Path(f"{RANKING_BASE_FOLDER}/msmarco_psg_bm25_rankings.trec_2019.tsv"))

def get_trec_2019_bm25_ranking_local():
    return Ranking.from_file(Path(f"/home/catalinlup/MyWorkspace/MasterThesis/datasets/run_files/msmarco_psg_bm25_rankings.trec_2019.tsv"))

def get_trec_2019_bm25_ranking_local_sample():
    return Ranking.from_file(Path(f"/home/catalinlup/MyWorkspace/MasterThesis/datasets/run_files/msmarco_psg_bm25_rankings.trec_2019.sample.small.tsv"))

def get_trec_2020_bm25_ranking_local():
    return Ranking.from_file(Path(f"/home/catalinlup/MyWorkspace/MasterThesis/datasets/run_files/msmarco_psg_bm25_rankings.trec_2020.tsv"))

def get_msmarco_dev_sample_bm25_ranking():
    return Ranking.from_file(Path(f"{RANKING_BASE_FOLDER}/msmarco_psg_bm25_rankings.dev.sample.tsv"))

def get_msmarco_dev_sample_bm25_ranking_local():
    return Ranking.from_file(Path(f"/home/catalinlup/MyWorkspace/MasterThesis/datasets/reranking_dev_sample_original/msmarco_psg_bm25_rankings.dev.sample.tsv"))

def get_msmarco_dev_sample_bm25_ranking_local_mock():
    return Ranking.from_file(Path(f"/home/catalinlup/MyWorkspace/MasterThesis/datasets/run_files/msmarco_psg_bm25_rankings.dev.sample.small.tsv"))

def get_msmarco_dev_colbert_sample_bm25_ranking_local():
    return Ranking.from_file(Path(f"/home/catalinlup/MyWorkspace/MasterThesis/datasets/run_files/msmarco_psg_bm25_rankings.dev.colbert_sample.tsv"))