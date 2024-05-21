# Wavelet Tree Transformer: Multi-Head Attention with Freguency Selective Representation and Interaction for Remote Sensing Object Detection 
### Jiahao Pan, Chu He, Wei Huang, Jidong Cao, and Ming Tong



main code in my paper:《Wavelet Tree Transformer: Multi-Head Attention with Freguency Selective Representation and Interaction for Remote Sensing Object Detection》.

    The main innovative modification of the article is in WaveTreeHeadAttention:
        1. channel lifting scheme module.
        2. channel lifting scheme in lifting.py including model importance for head att: three methods.
        3. wavelet tree selection use in key/value.
        4. idwt_by_tree



## Usage

Environment:

- Python 3.8.16
- torch 1.13.0+torchaudio 0.13.0+torchvision 0.14.0
- or: torch 1.12.1+cu113+torchaudio 0.12.1+cu113+torchvision 0.13.1+cu113
- 
HBB:
- mmcv-full 1.7.1
- mmdet 2.26.0
- 
OBB:
- mmcv-full                     1.7.1
- mmdet                         2.28.2
- mmrotate                      0.3.4   



## Citation

If this repo is useful for your research, please consider citation


## Statement
The authors would like to thank the editor and the anonymous reviewers for their valuable and constructive comments and suggestions, which have greatly contributed to improving the quality of this article.

## References






