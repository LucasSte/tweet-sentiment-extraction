# Tweet Sentiment Extraction

Jupyter Notebook for the tweet-sentiment-extraction [Kaggle](https://www.kaggle.com/c/tweet-sentiment-extraction) competition.

## Brief comptetition description

The competition aims at picking the phrase that reflects the labeled sentiment on a tweet. A tweet can be positive, negative or neutral, but identifying the words that best highlight those classification is an intricate task.

For instance, given a tweet "My dog is awesome", labeled as positive, we should develop an algorithm to choose "is awesome" as the best phrase to reflect that sentiment.

## roBERTa Model

[This](https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705) notebook inspired our work as a way to learn state-of-the art natural language processing techniques. Many sections of our notebook are similiar to the aforementioned one. We intended to dive into fine-tuning the roBERTa model for this competition.

We used a pretrained roBERTa base model with custom layers to adapat its output for our problem. The encoded tweet goes through roBERTa. Next, we use dropout to avoid overfitting and three convolutional layers with 128, 64, and 32 filters each. After that, we use a regular densely-connected NN layer with a LeakyReLu activation.  Finally, we flatten the result and apply a softmax activation to convert a real vector to a distribuiton of probabilities.

The model's outputs  represent the start and the end position of the picked sentiment phrase according to the tokenized tweet.

We optimized the model by using [Adam algorithm](https://keras.io/api/optimizers/adam/) and the Categorical Crossentropy loss function. Our dataset is the one available at the competition website.


```
Kaggle private score: 0.70511
Kaggle public score:  0.70428
```

 ## Files

Our main file is the ```roBERTa Model.ipynb```. It contains the code to train our neural network and test it.

We wanted to play with the model and created another jupyter notebook ```robertaClassifier.ipynb``` to classify a tweet as negative, positve or neutral without the burden of finding the sentiment words. The model only receives as input a tweet and outputs its emotion. It has two convolutional layers and two fully connected ones after roBERTa.

Moreover, in ```SentimentClassification.ipynb``` we built the complete sentiment pipeline. Given a tweet, our roBERTa classifier will tell if it is positive, negative or neutral. Then, our roBERTa model will find the words that highlight this sentiment.

Check our releases on GitHub for the trained models.


