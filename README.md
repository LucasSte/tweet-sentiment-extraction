# Tweet Sentiment Extraction

Jupyter Notebook for tweet-sentiment-extraction [kaggle](https://www.kaggle.com/c/tweet-sentiment-extraction) competition. The 

## Brief comptetition description

The competition aims at picking the phrase that reflects the labeled sentiment on a tweet. A tweet can be positive, negative or neutral, but identifying the words that best highlight those classifications is an intricate task.

For instance, given a tweet "My dog is awesome", labeled as positive, we should develop an algorithm to choose "is awesome" as the best phrase to reflect that sentiment.

## Build roBERTa Model

[This](https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705) notebook inspired our work as a way to learn state-of-the art netural language techniques. Many sections of our notebook are similiar to the aforementioned one. We intended to dive into fine-tuning the roBERTa model for this competition.

We used a pretrained roBERTa base model with custom layers to adpat its output for our problem. The encoded tweet goes through roBERTa that outputs the variable ```x```. Next, we use dropout to avoid overfitting and three conlutional layers with 128, 64, and 32 filters, each. After that, we use a regular densely-connected NN layer with a LeakyReLu activation.  Finally, we flatten the result and we apply a softmax activation to convert a real vector to a distribuiton of probabilities.

The outputs ```x1``` and ```x2``` represent the start and the end position of the picked sentiment phrase according to the tokenized tweet.

After all, we use a Fine Tuning technique to optimize the model by using [Adam algorithm](https://keras.io/api/optimizers/adam/) and the Categorical Crossentropy loss function.

```python
def build_model():
    ids = tf.keras.layers.Input((max_length,), dtype=tf.int32)
    att = tf.keras.layers.Input((max_length,), dtype=tf.int32)
    tok = tf.keras.layers.Input((max_length,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(path+'/config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(path+'/pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) # dropout randomly sets input units to 0 with a frequency of 10%
    x1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='same')(x1) # it creates kernel that is convolved with the layer input over a single spatial dimension
    x1 = tf.keras.layers.LeakyReLU()(x1) # it allows a small gradient when the unit is not active
    x1 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same')(x1)
    x1 = tf.keras.layers.Dense(units=1)(x1) # just a regular densely-connected NN layer
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Flatten()(x1) # it flattens the input and does not affect the batch size
    x1 = tf.keras.layers.Activation('softmax')(x1) # it converts a real vector to a vector of categorical probabilities
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0])
    x2 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(units=1)(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # Fine tuning
    model.compile(loss='categorical_crossentropy', optimizer=optimizer) # computes the crossentropy loss between the labels and predictions

    return model

model = build_model()
model.summary()
```
```



## Results

```
Kaggle private score: 0.70511
Kaggle public score:  0.70428
```

 ## Files

Our main file is the ```roBERTa Model.ipynb```. It contains the code to train our neural network and test it.

We wanted to play with the model and created another jupyter notebook ```robertaClassifier.ipynb``` to classify a tweet as negative, positve or neutral without the burden of finding the sentiment words. The model only receives as input a tweet and outputs its emotion.

Moreover, in ```SentimentClassification.ipynb``` we built the complete sentiment pipeling. Given a tweet, out roBERTa classifier will tell if it is positive, negative or neutral. Then, our roBERTa model will find the words that highlight this sentiment.
