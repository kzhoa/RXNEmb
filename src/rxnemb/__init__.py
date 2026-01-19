import warnings

# Suppress stupid UserWarnings from torch_geometric
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

from .core.rxn_emb import RXNEMB
