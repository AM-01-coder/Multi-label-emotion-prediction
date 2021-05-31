
from flask import Flask, render_template,request
import pickle
import numpy as np


#create an app and load models and data
app= Flask(__name__)

Tfidf_vectorizer = pickle.load(open('vectorizer_word_unigram.pkl', 'rb'))

    
with open("RF_trained_model", "rb") as f:
    Random_Forest_classifier = pickle.load(f)
output_of_training_data = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 
                           'Love','Optimism', 'Pessimism', 'Sadness', 'Surprise', 'Trust']
responses_columns = output_of_training_data
Target_categories= np.array(responses_columns)

@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
   
    string = request.form.values()
    mylist = list(set(string))
    mylist = ", ".join(mylist)
    
    
    unseen_feature_vectors = request.form.values()
    unseen_feature_vectors = Tfidf_vectorizer.transform(unseen_feature_vectors)
    unseen_feature_vectors = unseen_feature_vectors.todense()

    predicted_category = Random_Forest_classifier.predict(unseen_feature_vectors)
    predict_values = np.array(predicted_category)
    result = predict_values.flatten() 
    
    
    categorical_array = []
    counter=0
    for x in result:
        if x==1:
            categorical_array.append(Target_categories[counter])
        counter+=1
    if not categorical_array:
        categorical_array.append("Neutral")
   
        
    separator = ", "    
    separator = separator.join(categorical_array)       
    
    return render_template('Home.html', text1=mylist, prediction_text1=separator)



if __name__ == "__main__":
    app.run(debug=True)