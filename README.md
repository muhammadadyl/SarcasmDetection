# Twitter Sarcasm Detection Using Transformers


This repository is based on the [Transformers](https://github.com/huggingface/transformers) library by HuggingFace. It is intended as a starting point for anyone who wishes to use Transformer models in text classification tasks.

Table of contents
=================

<!--ts-->
   * [Setup](#Setup)
      * [With Conda](#with-conda)
   * [Usage](#usage)
      * [Twitter Sarcasm Dataset](#twitter-sarcasm-dataset)
      * [Current Pretrained Models](#current-pretrained-models)
      * [Custom Datasets](#custom-datasets)
      * [Evaluation Metrics](#evaluation-metrics)
   * [Acknowledgements](#acknowledgements)
<!--te-->

## Setup

### With Conda

1. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/distribution/)
2. Create a new virtual environment and install packages.  
`conda create -n transformers python pandas tqdm jupyter`  
`conda activate transformers`  
If using cuda:  
  `conda install pytorch cudatoolkit=10.0 -c pytorch`  
else:  
  `conda install pytorch cpuonly -c pytorch`  
`conda install -c anaconda scipy`  
`conda install -c anaconda scikit-learn`  
`pip install transformers` or download source code from [Transformers](https://github.com/huggingface/transformers)*
3. Clone repo.
`git clone https://github.com/muhammadadyl/SarcasmDetection.git`

*Important if you wanted to run DistilRoBERTa (soft release)

## Usage

### Twitter Sarcasm Dataset

If you are doing it manually;

Files are already available for use in `data/` folder with name `train.csv` and `test.csv`.

Once the download is complete, you can run the [data_prep_sarcasm.ipynb](data_prep_sarcasm.ipynb) notebook to get the data ready for training.

Finally, you can run the [run_model.ipynb](run_model.ipynb) notebook to fine-tune a Transformer model on the Twitter Dataset and evaluate the results.

### Current Pretrained Models

The table below shows the currently available model types and their models. You can use any of these by setting the `model_type` and `model_name` in the `args` dictionary. For more information about pretrained models, see [HuggingFace docs](https://huggingface.co/pytorch-transformers/pretrained_models.html).

| Architecture        | Model Type           | Model Name  | Details  |
| :------------- |:----------| :-------------| :-----------------------------|
| BERT      | bert | bert-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters.<br>Trained on cased English text. |
| XLNet      | xlnet | xlnet-base-cased | 12-layer, 768-hidden, 12-heads, 110M parameters. <br>XLNet English model |
| RoBERTa      | roberta | roberta-base | 125M parameters <br>RoBERTa using the BERT-base architecture |
| DistilBERT   | distilbert | distilbert-base-uncased | 6-layer, 768-hidden, 12-heads, 66M parameters <br>DistilBERT uncased base model |
| DistilRoBERTa      | distilroberta | distilroberta-base | 6-layer, 768-hidden, 12-heads, 82M parameters <br>DistilRoBERTa-base model. |

Note: DistilRoBERTa is in a soft release as of the day this repo published, to run this model you need to explicitly install Transformer library from Hugging Face's Repository. Installing through pip won't work here.

### Custom Datasets

When working with your own datasets, you can create a script/notebook similar to [data_prep_sarcasm.ipynb](data_prep_sarcasm.ipynb) that will convert the dataset to a Transformer ready format.

The data needs to be in `tsv` format, with four columns, and no header.

This is the required structure.

- `guid`: An ID for the row.
- `label`: The label for the row (should be an int).
- `alpha`: A column of the same letter for all rows. Not used in classification but still expected by the `DataProcessor`.
- `text`: The sentence or sequence of text.

### Evaluation Metrics

The evaluation process in the [run_model.ipynb](run_model.ipynb) notebook outputs the confusion matrix, and the Matthews correlation coefficient. If you wish to add any more evaluation metrics, simply edit the `get_eval_reports()` function in the notebook. This function takes the predictions and the ground truth labels as parameters, therefore you can add any custom metrics calculations to the function as required.

## Acknowledgements

None of this would have been possible without the hard work by the HuggingFace team in developing the [Transformers](https://github.com/huggingface/transformers) library.
