# PetsRetrieval
This is a project for 2020 Google Machine Learning Winter Camp.

Poster Link: https://docs.google.com/presentation/d/1jT8YBh6DwS_ZNbghAYk7DAUWY8U1U8-RDNq0uNt3Yio/edit?usp=sharing

Presentation Link: https://docs.google.com/presentation/d/1ayiKFiWZtsk9VpnvKVFDKMUyipq1vJJdgYVphkdgU7c/edit?usp=sharing
****

## Instructions
Please install some requirements in conda first befor running models.

`conda env create -f envTrain.yaml`
## Http Server
Http server is build with python flask, uwsgi and nginx. Please enter server fold first

`cd server` 
#### initStart.py
The entrance of main flask file is initStart.py. run `python initStart.py`
then you can test your server in http://localhost:6009
#### image_search.py
Call the trained model to generate a vector from the image. This class is called when 
the server receives a request for image retrieval.
#### ImageCaption.py
Call the trained model to generate a sentence from the image. This class is called when 
the server receives a request for image retrieval.
#### TextSearch.py
Call the database to pattern the request query text. This class is called when 
the server receives a request for text retrieval.
#### retrieval_post.py
The main file of retrieval logic processing. 
When a retrieval request is received, the corresponding file is processed and other file processing requests are called here.

## Image Retrieval
The main fold is in server/RetrievalPet. Please `cd server` first.
#### train a model with softmax
```
python -m RetrievalPet.examples.train_pet_softmax EXPORT_DIR=petModel -d pet
```
Hard Mining wiht softmax result is in server/RetrievalPet/hardMining.ipynb
After hard posivivate mining, you can train your model with DML metric.
#### train a model with triplet
The gem pooling method is from [gem](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
 .
```
python -m RetrievalPet.examples.train_pet EXPORT_DIR=petModel -d pet -p gem
```
#### train a model with caption
If you want to train a model with caption, you can run the following.
Note: please do image caption first.
```
python -m RetrievalPet.examples.train_pet EXPORT_DIR=petModel \
                                -d pet --use-caption -p gem
```
#### Test for a given image
If want to test with a extracted embeddings, please run
```
python  -um RetrievalPet.examples.test_extract2  
        --network-path  EXPORT_DIR\=petModel/... --use-caption
 ```
#### Test with PCA
  ```
 python  -um RetrievalPet.examples.PCA
  ```
#### Test with DBA
  ```
 python  -um RetrievalPet.examples.DBA
  ```
#### Test with AQE
  ```
 python  -um RetrievalPet.examples.QE
  ```
#### Test with IR2
   The CDVS extractor is from [cdvs](https://github.com/WendyDong/ImageRetrieval_DF_CDVS).
  ```
 python  -um RetrievalPet.examples.IR2
  ```
