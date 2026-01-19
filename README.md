



![RXNEmb](static/head_banner.png)


![PyPI - Downloads](https://img.shields.io/pypi/dm/rxnemb?label=PyPI%20downloads&logo=pypi)
![Python version](https://img.shields.io/badge/python->=3.9-blue)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**RXNEmb** is a novel reaction-level embedding descriptor generated via the pre-trained model [**RXNGraphormer**](https://github.com/licheng-xu-echo/RXNGraphormer). It captures chemical bond formation and cleavage patterns, enabling data-driven reaction classification, mechanistic interpretation, and reaction space visualization.


# 🚀 Overview

The preprint paper is available at [https://arxiv.org/pdf/2601.03689](https://arxiv.org/pdf/2601.03689) .


# 🔧 Installation 

```bash
pip install rxnemb
```


# 💡  Generate RXNEmb Descriptor

## Basic Usage
**Example 1: Generate reaction embeddings from SMILES list**

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


## Use Customized Model

If you want to train your own model from scratch, please refer to the instructions in the  [**RXNGraphormer**](https://github.com/licheng-xu-echo/RXNGraphormer) repository.

After training is done, you can load the model by specifying the path to the directory containing your model weights and configuration file:

```python
from rxnemb import RXNEMB

generator = RXNEMB('/path/to/your/own/model')
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



