
  

  

# How To Run Server Application

  

  

# To Run Application:

  
Install environment 

    conda activate /Users/learn/Desktop/Projects/machine-translation/server/venv

Activate environment

  

conda activate /Users/learn/Desktop/Projects/machine-translation/machine-learning/venv

Install kernel

  

python -m ipykernel install --user --name machine-translation --display-name "machine-translation"

Open Jupyter

  
  

jupyter notebook

  

  

# Folder Structure

  

  

## train

  

  

### lstm:

  

  

index.ipynb: train lstm model that can translate from hindi to english **data_size**: determines how many sentences will be used for training model is trained and saved in disk Performance metrics like **BLEU , ROUGE, Cosine Similarity** are calculated to determine the quality of the translation

  

  

## test

  

  

### lstm:

  

  

index.ipynb: uses trained model (from the disk) to calculate performance metrics like **BLEU , ROUGE, Cosine Similarity** in test sentences

  

  

## model

  

  

### lstm:

  

  

consist of all the trained model along with their dictionaries. You can load any of the model from here

  

  

## data

  

  

iit_dataset.py: loads train and test dataset from iit archive

  

  

## analytics

  

  

index.ipynb: bar chart of performance metric for trained model analytics.json: BLEU, ROUGE score for model for different size are save in this file

  

  

## utils

  

  

consist of helper functions

  

  

# Keywords

  

  

machine translation , machine learning, NLP, , tensorflow

  

  

## Notes

  

  

For futher queries, please contact me via email at : [asif01050105@gmail.com](mailto:asif01050105@gmail.com  "mailto:asif01050105@gmail.com")

  

[1421015@iub.edu.bd](mailto:1421015@iub.edu.bd  "mailto:1421015@iub.edu.bd")

  

  

**Asif Mahmud**
