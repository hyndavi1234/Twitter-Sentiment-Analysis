# Twitter-Sentiment-Analysis
Design a machine learning model to categorize the sentiment of the tweets. This has been implemented as part of the course project.

In NLP research, sentiment analysis is an active area of interest. It deals with predicting the opinion of a given text/sentence. It has many names such as Opinion Mining, Sentiment Mining, and Subjectivity Analysis. Sentiment Analysis helps companies and organizations in making data-driven decisions. It has a lot of applications such as chatbot, product reviews, aspect reviews, and so on. Here, we focused on sentence level sentiment analysis. It involves understanding the sentence word by word to obtain the semantic meaning of a sentence.

## Overview
The goal is to design a machine learning model to classify the sentiment of the tweets into three classes labels positive, negative, and neutral. we worked on two different datasets  that contains the tweets during a presidential debate between Obama and Romney few years ago. The performance of the models is evaluated based on accuracy, precision, recall and F1 score of the positive, neutral and negative classes. We were successfully able to design the model, perform training according to the nature of data to predict the class labels for the provided test tweets.

## Techniques
Following are the various techniques applied to design and predict the model.

### Data Preprocessing
Data cleaning and pre-processing is the crucial foremost step in any data driven applications.
- Removed rows with missing tweets and labels. 
- Removed HTML tags, URLs, Hashtags (#), Mentions (@), punctuations and stop words.
- Decontracted the text, performed lemmatization.
- Removed classes other than –1/0/1 and then label encoded the ‘class’

### Data Analysis
Values of class {-1, 0, 1} are for negative, neutral, and positive classes respectively.

![github image](https://user-images.githubusercontent.com/34919619/211905884-573695e0-d337-485c-9399-3f9fccc45bd5.png)

### Feature Engineering
Applied different vectorizations on the text data suchh as Bag-of-words, TF-IDF, pretrained Word2Vec (200 Dimensions, 300 Dimensions), TF-IDF with Word2Vec 300 Dim. 

Bag of Words only incorporates the frequency count of each word whereas TF-IDF reflects how important a word is to a document in a corpus. We also used W2V-300 dimensional embeddings as they store more information about the semantic representation of the word when compared to W2V-200 dimensional embeddings.

### Modelling
We have trained many classifiers to understand their classification behavior by assessing their metrics to select the most accurate model. We used XGB classifier, Logistic Classifier, Random Forests, Multinomial  Naive Bayes, and a Voting Classifier. All the above classifiers are included for constructing Voting Classifier. While testing, there was a high variability in the predictions produced by the models. So, in order to  balance, we used a voting classifier. In addition, we tuned the hyper-parameters for all these models.

While experimenting with various text vectorizations, we discovered that the number of features is  greater than the total number of tweets; consequently, we adjusted a few parameters, such as ‘min_df’ (cut-off) in TF-IDF, to perform better (min_df = 2 provided good results). Additionally, an attempt  was made to reduce the vector size by using K-Means clustering. Experimented to discover the optimal  cluster (tested between ~300 to 400 clusters) using the k-fold analysis, but the silhouette score was poor.

Furthermore, we could not locate the elbow point when plotting the silhouette index versus the number  of clusters using the Elbow method, hence we did not proceed with K-Means. To increase accuracy of existing models, we have included Vader model SentimentIntensityAnalyzer  polarity scores as new features to each document feature vector. We only trained the Multinomial Naive  Bayes classifier for TF-IDF because this model cannot handle negative integers, which existed in other  vectorizers. As we included the Multinomial NB in Voting classifier, we first normalized the input vectors  by using min-max normalization before training the model. As Voting classifier is an ensemble model that  combines multiple classifiers, it outperformed in comparison to individual model performance. We have  also tested the BERT pre-trained model and word embeddings, but its performance was not as expected  even after many epochs due to less data. Having sufficient training data is quite valuable here.   

To handle the disparity of classes in Romney data, we added the class weight parameter as balanced to  predict the appropriate class labels without any bias. It resulted in good class ratios, as the balanced mode  automatically adjusts weights according to class frequency proportions.   

We used 10-fold stratified cross-validation while training and observed that bag-of-words models  significantly underperformed with all the other methods. Hence, we decided to eliminate BoW  vectorization.

## Experiment Outcomes
The highlighted classifiers are the final models used for corresponding Obama, Romney datasets.

Following are the evaluation metrics captured for all the models except Voting Classifier:
![github image](https://user-images.githubusercontent.com/34919619/211873449-643c85c1-fbff-4685-aa23-c00ff1bb83f6.png)

For Voting Classifier, following are the obtained results for both datasets:
![github image](https://user-images.githubusercontent.com/34919619/211905888-ff140f7e-aa64-4b96-b2c6-51ee83aed1fa.png)

## Conclusion
Obama Dataset: Using the Voting classifier with W2V-300 dimensions, we obtained the maximum accuracy of 0.64 and a good balance of precision, recall, and F-1 score. As a result, we finalized it as the model for the dataset.

Romney Dataset: There is a class imbalance since more negative labels exist than positive ones. This results in a skewed model that labels the majority of test tweets as negative. We had to analyze several metrics to determine which model best suited the data. Although there are other models with an accuracy of 0.64, we chose Logistic regression with W2V – 300 dimensions as it has high recall and F-1 score for the positive class.

Utilizing multiple ML models enabled us to comprehend how each classifier responds to data and respective vectorization techniques. Understanding the type of data to do pre-processing, analyzing the proportion of classes, identifying different vectorization approaches, and how each affects the model's behavior are the key takeaways from this research study. If the classes are highly imbalanced, even the most outstanding models, such as Transformers, cannot capture the correct data patterns. In such cases, we can utilize strategies such as Under sampling and Over sampling (SMOTE). Furthermore, the amount of data plays a significant influence in accurately predicting class labels. To improve the outcomes, we could utilize Neural Networks to use techniques like transfer learning, adding custom features, and 
effectively using vectorization techniques.






