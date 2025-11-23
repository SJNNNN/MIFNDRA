# MIFNDRA
MIFNDRA: an innovative knowledge-enhanced multimodal fusion and graph learning framework for predicting drug resistance-related ncRNAs
## Requirements
  * python==3.7
  * dgl==0.6.1
  * networkx==2.5
  * numpy==1.16.6
  * scikit-learn==0.20.3
  * pytorch==1.5.0
  * tqdm==4.15.0

## File
### data
  The data files needed to run the model.
  * disease semantic similarity matrix 1.txt and disease semantic similarity matrix 2.txt: Two kinds of disease semantic similarity
  * ncRNA_Functional_Similarity_Matrix.txt: MiRNA functional similarity
  * ncrna_rug_index.txt:Validated ncRNA-drug resistance associations

### code
  * eval.py: The startup code of the program
  * train.py: Train the model
  * model.py: Structure of the model
  * utils.py: Methods of data processing
 
## Usage
  * download code and data
  * execute ```python eval.py```
