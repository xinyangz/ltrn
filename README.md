# Minimally Supervised Structure Rich Text Categorization by Learning on Text-Rich Networks

# Paper
Our paper can be accessed [here](https://arxiv.org/abs/2102.11479).

# Running the Experiments
## Requirements
The code requires Python 3.7+ and the HuggingFace Transformers library `transformers==4.1.0`. The detailed requirements can be found in `requirements.txt`. Note that specific versions of `torch_scatter`, `torch_sparse`, `torch` might be needed to work with different Cuda versions.

## Steps to Run the Experiments

### Download data

The data can be accessed through [Dropbox](https://www.dropbox.com/sh/7vuglt3fd7m12a9/AAB3AdVeLsEgi-8UjqiIKtXMa?dl=0).

### Run the training script
Edit the training scripts `run_amazon.sh` and `run_books.sh` to specify path to data and the output.
Then execute the scripts to run the experiments.

### Running on custom datasets
Please follow the given datasets to format your data. Then create a training script to run the experiments.
# Citation
Please cite the following paper if you found our dataset or framework useful. Thanks!

```bibtex
@inproceedings{zhang2021ltrn,
  author = {Zhang, Xinyang and Zhang, Chenwei and Dong, Luna Xin and Shang, Jingbo and Han, Jiawei},
  title = {Minimally Supervised Structure Rich Text Categorization by Learning on Text-Rich Networks},
  year = {2021},
  booktitle = {Proceedings of The Web Conference 2021},
  location = {Ljubljana, Slovenia},
  series = {WWW '21}
}
```

