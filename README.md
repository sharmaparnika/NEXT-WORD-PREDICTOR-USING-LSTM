# NEXT-WORD-PREDICTOR-USING-LSTM

#### INTRODUCTION:

How does the keyboard on your phone know what you would like to type next? Language prediction is a Natural Language Processing - NLP application concerned with predicting the text given in the preceding text. Auto-complete or suggested responses are popular types of language prediction. The first step towards language prediction is the selection of a language model.

Vanishing gradient descend is a problem faced by neural networks when we go for backpropagation. It has a huge effect and the weight update process is widely affected and the model became useless. So, we used LSTM which has a hidden state and a memory cell with three gates that are forgotten, read, and input gate.

![image](https://user-images.githubusercontent.com/73773202/163566623-a43ec35c-5c87-4a37-a457-d07dc3de0589.png)

•	The forget gate is mainly used to get good control of what information needs to be removed which isn’t necessary. 

•	Input gate makes sure that newer information is added to the cell and output makes sure what parts of the cell are output to the next hidden state. 

•	The sigmoid function used in each gate equation makes sure we can bring down the value to either a 0 or 1.
 
#### Application of LSTM:

Predicting the next word is a neural application that uses Recurrent neural networks. Since basic recurrent neural networks have a lot of flaws we go for LSTM. Here we can make sure of having longer memory of what words are important with help of those three gates we saw earlier. 

The following diagram tells us exactly what we are trying to deal with. What could be the next word? We will build a neural model to predict this.

![image](https://user-images.githubusercontent.com/73773202/163566705-9fe220bf-470a-4f93-b460-48b4a1839932.png)

## Requirements:
* Python 3
* Numpy
* tensorflow
* pickle
* os
* text file
* Natural Language Processing (Bag of Words)

## Building Our Predictor:


Importing the required libraries:

    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    import pickle
    import numpy as np
    import os

Uploading the text file

    from google.colab import files
    uploaded = files.upload()

Loading and pre-processing the data:

    file = open("Art_of_war.txt", "r", encoding = "utf8")

    #store file in list
    lines = []
    for i in file:
    lines.append(i)

    #Convert list to string
    data = ""
    for i in lines:
     data = ' '. join(lines) 

    #replace unnecessary stuff with space
    data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')  #new line, carriage return, unicode character --> replace by     space

    #remove unnecessary spaces 
    data = data.split()
    data = ' '.join(data)
    data[:500]


    len(data)

Applying Tokenization
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])

    # saving the tokenizer for predict function
    pickle.dump(tokenizer, open('token.pkl', 'wb'))

    sequence_data = tokenizer.texts_to_sequences([data])[0]
    sequence_data[:15]

    len(sequence_data)

    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)

    sequences = []

    for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)
    
    print("The Length of sequences are: ", len(sequences))
    sequences = np.array(sequences)
    sequences[:10]

    X = []
    y = []

    for i in sequences:
    X.append(i[0:3])
    y.append(i[3])
    
    X = np.array(X)
    y = np.array(y)

    print("Data: ", X[:10])
    print("Response: ", y[:10])

    y = to_categorical(y, num_classes=vocab_size)
    y[:5]

Creating the model

    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=3))
    model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(1000))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(vocab_size, activation="softmax"))

    model.summary()

Plotting the model

    from tensorflow import keras
    from keras.utils.vis_utils import plot_model

    keras.utils.plot_model(model, to_file='plot.png', show_layer_names=True)

 

Training the model

    from tensorflow.keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
    model.fit(X, y, epochs=70, batch_size=64, callbacks=[checkpoint])

Making predictions
    
    Let's predict
    from tensorflow.keras.models import load_model
    import numpy as np
    import pickle

    # Load the model and tokenizer
    model = load_model('next_words.h5')
    tokenizer = pickle.load(open('token.pkl', 'rb'))

    def Predict_Next_Words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""
  
    for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break
  
    print(predicted_word)
    return predicted_word

    while(True):
    text = input("Enter your line: ")
  
    if text == "0":
       print("Execution completed.....")
      break
  
    else:
      try:
          text = text.split(" ")
          text = text[-3:]
          print(text)
        
          Predict_Next_Words(model, tokenizer, text)
          
      except Exception as e:
        print("Error occurred: ",e)
        continue
     
     
  Lets check our Predictor:
  

https://user-images.githubusercontent.com/73773202/163569350-945cb5ac-5829-4639-be47-57ca1dcb2069.mp4

