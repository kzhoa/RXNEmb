# 🚀 Overview

**RXNEmb** is a novel reaction-level embedding descriptor generated via the pre-trained model [**RXNGraphormer**](https://github.com/licheng-xu-echo/RXNGraphormer). It captures chemical bond formation and cleavage patterns, enabling data-driven reaction classification, mechanistic interpretation, and reaction space visualization.

The preprint paper is available at [https://arxiv.org/pdf/2601.03689](https://arxiv.org/pdf/2601.03689) .


# 🔧 Installation 

```bash
pip install rxnemb
```


# 💡  Generate RXNEmb Descriptor

**Example: Generate reaction embeddings from SMILES list**

```python
from rxnemb import RXNEMB

generator = RXNEMB()

# prepare reaction SMILES
# here are some examples
rxn_smiles_lst = [
    "O=C(O)c1ccccc1.[Cl-]>>O=C(Cl)c1ccccc1",  
    "C1CCOC1.C1CCOC1.O.O=C(O)C(Br)c1ccccc1>>OCC(Br)c1ccccc1",
    # ... more 
]

# output
rxn_emb = generator.gen_rxn_emb(rxn_smiles_lst)
print(rxn_emb.shape) # torch.Size([2, 768])
```


# 📚 Citation

If you use RXNEmb in your research, please cite:

```bash
@misc{RXNEmb_liu2026,
      title={A Pre-trained Reaction Embedding Descriptor Capturing Bond Transformation Patterns}, 
      author={Weiqi Liu and Fenglei Cao and Yuan Qi and Li-Cheng Xu},
      date = {2026-01-07},
      eprint={2601.03689},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi = {10.48550/arXiv.2601.03689},
      url={https://arxiv.org/abs/2601.03689}, 
      pubstate = {prepublished},
}
```



