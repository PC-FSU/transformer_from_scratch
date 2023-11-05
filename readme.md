# Attention is all you need

I try to follow this [tutorial](https://youtu.be/ISNdQcPhsts) to implement "Attention is all you need" [paper](https://arxiv.org/abs/1706.03762). For original work please refer to the link provided above. 

## Project Files

- [config.py](config.py): Contains model configuration and utility functions for saving/loading model file paths.

- [dataset.py](dataset.py): Provides functions to return training and validation dataset objects.

- [train.py](train.py): This script is responsible for model training and validation. It downloads an English-to-Italian translation dataset from Huggingface and utilizes the Transformer model defined in [model.py](model.py). Additionally, it provides the option to use a pre-trained model for machine translation tasks.

- [model.py](model.py): Contains the components required to build a Transformer model.
