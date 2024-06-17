import sys
sys.path.append('../../')
from quantized_fast_forward.fast_forward.encoder import QueryEncoder
from pyserini.search.faiss import AggretrieverQueryEncoder, TctColBertQueryEncoder
import numpy as np
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import Run, RunConfig, ColBERTConfig



# Define AggRetriever Query Encoder
class FFAggretrieverQueryEncoder(QueryEncoder):
    def __init__(self, model_name, device="cpu"):
        self._enc = AggretrieverQueryEncoder(model_name, device=device)

    def encode(self, queries):
        return np.array([self._enc.encode(q) for q in queries])
    

# Define the TctColBERT Query Encoder
class FFTctColBertQueryEncoder(QueryEncoder):
    def __init__(self, model_name, device='cpu'):
        self._enc = TctColBertQueryEncoder(model_name, device=device)

    def encode(self, queries):
        return np.array([self._enc.encode(q) for q in queries])
    


# Define the  ColBERT Encoder
class FFColBertQueryEncoder(QueryEncoder):
    def __init__(self, model_path) -> None:
        config = ColBERTConfig()
        self.checkpoint = Checkpoint(model_path, config)
    
    def encode(self, queries):
      
        return self.checkpoint.queryFromText(queries).cpu().numpy()
    


print('Loading encoders...', flush=True)
AGG_RETRIEVER_ENCODER = FFAggretrieverQueryEncoder("castorini/aggretriever-cocondenser")
TCT_COLBERT_ENCODER = FFTctColBertQueryEncoder("castorini/tct_colbert-msmarco")
try:
    COLBERT_ENCODER = FFColBertQueryEncoder("/scratch/clupau/models/colbertv2.0")
except:
    print('Could not load Colbert Encoder')
    COLBERT_ENCODER = None
print('Encoders loaded.', flush=True)