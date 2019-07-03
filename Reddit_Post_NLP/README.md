
### Workflow:

- [Problem Statement](notebooks/Classifying_Posts.ipynb#Problem Statement)
- Data Gathering
- Data Exploration
- Pre-processing and Modeling
- Model Evaluation
- Concluding Remarks


### Problem Statment

<img src="./images/TVshowsMovieswordle2b.png" alt="drawing" width="800"/>

   
  For anyone familiar with television shows or movies, it is easy to identify if the Office is a TV show or not. Unlike humans, a computer cannot easily distinguish words that are associated with either a movie or TV show, or how to categorize them. However, with the appropriate training, a computer should be able to identify such words as a human does.

 This scenario is similar to the problem at hand. Let's assume that a start-up is attempting to launch a new product that can identify which words are associated with a TV show or a movie. That is, the problem at hand is a classification problem: is a word (or set of words) related to a tv show or not.

 Now assume that the project manager communicates this to you, as the only data scientist on the team (assuming it's a very small start-up), and assigns you the responsibility of training a model for this task. However, she does not provide you with any data, a key component of model building. After an extensive google search, you do not find any appropriate structured data for the task. You, therefore, resort to gathering unstructured data from Reddit. The first task is to gather the data, engineer it so that it can be transformed into structured data, and then develop, train the model, and evaluate the model. 


### Executive Summary

 In this project, we are faced with a classification task. The goal is to develop a model that can accurately place words in the right category. That is, we want to train a model that sees a word and correctly tells us if it is related to tv or not. 
   
   Since we do not have structured data at hand, we are left to gather unstructed text data. This kind of data never comes in 'clean'. It involves several iteartions of cleaning and parsing, which was the case in this project. 
   
   The data employed here was gathered from Reddit. We tackle the problem using movie and television subreddits. Given the restriction on the amount of data that can be retrieved per time, we collected 1231 movie posts and 2174 tv posts, making a total of 3405 posts. In more technical terms, we have 3045 documents of text data to analyze.
   
   We employ four algorithms and several models (due to hyperparameter tuning). 
   - Logistic Regression 
   - Naive Bayes (Multinomial)
   - Support Vector Classifier
   - Gradient Boosting Classifier
   
   The Logistic regression returned an 80% accuracy, the Naive Bayes model returned an accuracy of 89%, the Support Vector Classifier returned an accuracy of 91%, and the Gradient Boosting classifier an accuracy of 89%. 
   
   We eventually settle for two models: the Support Vector classifier and a Gradient Boosting model. These models tend to generalize fairly well. When we take these models to unseen data, we find that they perform just as well as they did when they were trained.  
   
   Using these models leaves us with the challenge of interpretability. That is, these kind of models, unlike logistic regressions and linear regressions, do not provide us with coefficients that can be easily interpreted. If our interest is to draw inference, then a logistic regression would be our best bet here. However, since the goal is not inference but accuracy, the lack of interpretation is not much of a problem. 
   
   The SVM classifier and GB model generalize well. However, the SVM classifier outperforms the GB classifier on the holdout set. Therefore, we settle on this model and recommend it for deployment for the task.  


### Data Gathering

   The data employed here was gathered from Reddit. We tackle the problem using movie and television subreddits. Given the restriction on the amount of data that can be retrieved per time, we collected 1231 movie posts and 2174 tv posts, making a total of 3405 posts. In more technical terms, we have 3045 documents of text data to analyze.

   The code for the data gathering process is not reported here. Please see the Jupyter Notebook titled data_gathering for the details of the code.

   After gathering the data, we randomly split the data into a traina and test split (or a holdout set). This was done to test our model's generalizability. When faced with new data, would our model perform just as well as it did when we trained it? We attempt to address this question in a convincing way.
    
   

### Data Exploration

   We identify missing data. For a number of cells, there is no text data even though we identify that users made comments related to a post. This activity can be seen in the `num_comments` column. This suggests that users posting comments may not have used words. Instead, they most likely posted a pic, gif, or even some other form of media. This is common in subreddits in general.
   We address this by conducting some feature engineering. By merging the title and the associated comments, we now have documents with text that can be used in our analysis. Furthermore, we binarize the `subreddit` column.Please see the data dictionary for a description of this binary variable.  

### Pre-Processing and Modeling

   To pre-process the data, we begin by placing our text data into a dataframe. This is done for two reasons: 1) to view our data more easily 2) to easily manipulate the data and prepare the data for modeling. The preprocessing is in five stages: 
   1. We remove html code from the text
   2. We use Regular Expressions (regex) to remove punctuations and other none-words
   3. We remove "stop-words". That is extremely common words which would be of little value in the analysis.
   4. The final step, which is transforming the data into values that a computer csan understand is done alongside our modeling, since we use a Pipeline and grid-search.
   
   We employ four algorithms and several models (due to hyperparameter tuning). We eventually settle for two models: the Support Vector classifier and a Gradient Boosting model. These models tend to generalize fairly well. 
   

### Model Evaluation
   
   We estimate our models accuracy, misclassification rate, and precision to evaluate their performnce. After an initial cross-validation, scoring and predicting on the test set, we took the model to unseen data. Both models, but the SVM in particular, performed just as well as they did on the training and test data. Unlike the Logistic regressiona and the Naive Bayes, there is no clear evidence of overfitting. 
   

### Concluding Remarks

   Using these models leaves us with the challenge of interpretability. That is, these kind of models, unlike logistic regressions and linear regressions, do not provide us with coefficients that can be easily interpreted. If our interest is to draw inference, then a logistic regression would be our best bet here. However, since the goal is not inference but accuracy, the lack of interpretation is not much of a problem. 
   
   The SVM classifier and GB model generalize well. However, the SVM classifier outperforms the GB classifier on the holdout set. Therefore, we settle on this model and recommend it for deployment for the task.     

   
   






```python

```
