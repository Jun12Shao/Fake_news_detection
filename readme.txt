1. Install: 
numpy, pandas, tqdm, sklearn, scipy, nltk, joblib

2. Data preprocessing:
In this work, we use sklearn.feature_extraction.text.TfidfVectorizer to generate features from the content: 'tile'+'author'+'title'.

3. Training Data was splitted into train/valid by 8/2. valid data was used for model selection. 
4. Model for fake news detection:
We adapted SVC(support vector machine classifier), Decision Tree, Random Forest and MLP classifier.


5.Evaluation metrics:
In order to evaluate the performance of our model, we take accuracy and F1-measure as evaluation metrics. 
Accuracy is the percentage of instances in the test set that the model has correctly classified. 
F-measure is a weighted combination of precision and recall.

6. Test Results:
Model      		Accuracy          	F1 Score 
SVC			63.56%			65.26%
Decision_Tree 		64.23%			66.21%
Random_Forest		72.87%			75.31%
MLP			63.73%			65.42%



