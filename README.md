# Tweet Sentiment Extraction

Jupyter Notebook for tweet-sentiment-extraction [kaggle](https://www.kaggle.com/c/tweet-sentiment-extraction) competition.

## Description

"My ridiculous dog is amazing." [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds. But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

Help build your skills in this important area with this broad dataset of tweets. Work on your technique to grab a top spot in this competition. What words in tweets support a positive, negative, or neutral sentiment? How can you help make that determination using machine learning tools?

In this competition we've extracted support phrases from Figure Eight's Data for Everyone platform. The dataset is titled Sentiment Analysis: Emotion in Text tweets with existing sentiment labels, used here under creative commons attribution 4.0. international licence. Your objective in this competition is to construct a model that can do the same - look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

## Dataset

| textID        | text                                             | selected_text                       | sentiment |
|:-------------:|:------------------------------------------------:|:-----------------------------------:|:---------:|
| cb774db0d1    | I`d have responded, if I were going              | I`d have responded, if I were going | neutral   |
| 549e992a42    | Sooo SAD I will miss you here in San Diego!!!    | Sooo SAD                            | negative  |
| 088c60f138    | my boss is bullying me...                        | bullying me                         | negative  |
| 9642c003ef    | what interview! leave me alone                   | leave me alone                      | negative  |
| 358bd9e861    | Sons of ****, why couldn`t they put them on t... | Sons of ****,                       | negative  |

## Build roBERTa Model

As we want to use a pretrained roBERTa base model, we will add custom layers to it in order to make our model appropriated to our problem. Hence, the first tokens are input into bert_model and its output is x as below. Also is worth mentioning that the previous output has a shape in a form of (batch_size, MAX_LEN, 768). Next, we drop out randomly sets input with frequency rate between 0% and 10%, in order to avoid overfitting. Then, we use a 1D convolutional layer three times with 128, 64, and 32 filters, respectively, in such a way that for the first two, we use a LeakyRelU activation layer. After the third one, we use a regular densely-connected NN layer, where N=1. Then, we use another LeakyRelU activation layer. Finally, we flatten the result and, after that, we apply a softmax activation layer to convert a real vector to a vector of categorical probabilities. Hence, the model output x1 for the start tokens indices and x2 for the end tokens indices.

After all, we use a Fine Tuning technique to optimize the model by using [Adam algorithm](https://keras.io/api/optimizers/adam/) and compile with categorical_crossentropy as the loss function.

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
All model checkpoint layers were used when initializing TFRobertaModel.

All the layers of TFRobertaModel were initialized from the model checkpoint at /content/drive/MyDrive/PO-240/Files/pretrained-roberta-base.h5.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFRobertaModel for predictions without further training.
Model: "functional_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_4 (InputLayer)            [(None, 96)]         0                                            
__________________________________________________________________________________________________
input_5 (InputLayer)            [(None, 96)]         0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            [(None, 96)]         0                                            
__________________________________________________________________________________________________
tf_roberta_model_1 (TFRobertaMo TFBaseModelOutputWit 124645632   input_4[0][0]                    
                                                                 input_5[0][0]                    
                                                                 input_6[0][0]                    
__________________________________________________________________________________________________
dropout_76 (Dropout)            (None, 96, 768)      0           tf_roberta_model_1[0][0]         
__________________________________________________________________________________________________
dropout_77 (Dropout)            (None, 96, 768)      0           tf_roberta_model_1[0][0]         
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 96, 128)      196736      dropout_76[0][0]                 
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 96, 128)      196736      dropout_77[0][0]                 
__________________________________________________________________________________________________
leaky_re_lu (LeakyReLU)         (None, 96, 128)      0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)       (None, 96, 128)      0           conv1d_5[0][0]                   
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 96, 64)       16448       leaky_re_lu[0][0]                
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 96, 64)       16448       leaky_re_lu_3[0][0]              
__________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)       (None, 96, 64)       0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)       (None, 96, 64)       0           conv1d_6[0][0]                   
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 96, 32)       4128        leaky_re_lu_1[0][0]              
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 96, 32)       4128        leaky_re_lu_4[0][0]              
__________________________________________________________________________________________________
dense (Dense)                   (None, 96, 1)        33          conv1d_4[0][0]                   
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 96, 1)        33          conv1d_7[0][0]                   
__________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)       (None, 96, 1)        0           dense[0][0]                      
__________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)       (None, 96, 1)        0           dense_1[0][0]                    
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 96)           0           leaky_re_lu_2[0][0]              
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 96)           0           leaky_re_lu_5[0][0]              
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 96)           0           flatten_2[0][0]                  
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 96)           0           flatten_3[0][0]                  
==================================================================================================
Total params: 125,080,322
Trainable params: 125,080,322
Non-trainable params: 0
```

## Results

```
>>>> FOLD 5 Jaccard = 0.8055669877520152
```

Read more in our jupyter notebook file!
