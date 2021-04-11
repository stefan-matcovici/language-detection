# language-detection
Small project to explore language detection techniques

### Dataset
Used [Wili2018 dataset](https://arxiv.org/pdf/1801.07779.pdf) with given train/test split

### Approaches report
|            	| accuracy 	| f1    	| precision 	| recall 	|
|------------	|----------	|-------	|-----------	|--------	|
| CLD2       	| 95.67    	| 91.83 	| 93.29     	| 90.63  	|
| langid     	| 97.28    	| 92.57 	| 93.12     	| 92.16  	|
| langdetect 	| 97.58    	| 92.62 	| 92.96     	| 92.44  	|
| ngram-nb 	    | 97.5    	| 97.53 	| 97.68     	| 97.5  	|
| biLSTM 	    | 97.57    	| 97.58 	| 97.62     	| 97.57  	|
| distil-bert 	| 99.39    	| 99.39 	| 99.39     	| 99.39  	|

### Implementation
1. [Available modules baseline](notebooks/0.baseline-available-modules.ipynb)
 
   Tested 3 solutions:
   * [CLD2](https://pypi.org/project/pycld2/)
   * [langid](https://pypi.org/project/langid/)
   * [langdetect](https://pypi.org/project/langdetect/)
---
2. [Ngram approach](notebooks/1.ngram.ipynb)
    * Combined ngram approach with TF-IDF and Naive Bayes
---
3. [RNN approach](notebooks/2.rnn.ipynb)
   * Implemented bidirectional LSTM
---
4. [Transformers approach](notebooks/3.transformers.ipynb)
   * Used pretrained `distilbert-base-multilingual-cased` and finetuned on trainset