from flask import Flask, request, render_template
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('E:/XLNNTN/news.csv')
X = df['text']
y = df['label']

def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# Random Forest Model
vectorizer_rf = CountVectorizer(preprocessor=preprocess, stop_words='english')
X_rf = vectorizer_rf.fit_transform(X)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y, test_size=0.2, random_state=42)
#Load model
clf_rf = joblib.load('random_forest_model.pkl')

y_pred = clf_rf.predict(X_test_rf)
rf_accuracy = accuracy_score(y_test_rf, y_pred)
print('Độ chính xác trên tập kiểm tra: {0}%'.format(round(rf_accuracy * 100, 0)))

# RNN Model

y_rnn = y.map({'FAKE': 0, 'REAL': 1})
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X, y_rnn, test_size=0.2, random_state=42)

tokenizer_rnn = Tokenizer(num_words=5000)
tokenizer_rnn.fit_on_texts(X_train_rnn)
    
# Chuyển đổi văn bản thành các chuỗi số nguyên
X_train_seq_rnn = tokenizer_rnn.texts_to_sequences(X_train_rnn)
X_test_seq_rnn = tokenizer_rnn.texts_to_sequences(X_test_rnn)


# Đặt độ dài cố định cho các chuỗi số nguyên
max_seq_length = 500
X_train_pad_rnn = pad_sequences(X_train_seq_rnn, maxlen=max_seq_length)
X_test_pad_rnn = pad_sequences(X_test_seq_rnn, maxlen=max_seq_length)


#inputs_rnn = Input(shape=(500,))
#embedding_rnn = Embedding(input_dim=5000, output_dim=32)(inputs_rnn)
#lstm_rnn = LSTM(units=64)(embedding_rnn)
#outputs_rnn = Dense(units=1, activation='sigmoid')(lstm_rnn)
#model_rnn = Model(inputs=inputs_rnn, outputs=outputs_rnn)
#model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model_rnn.fit(X_train_rnn, y_train_rnn, epochs=5, batch_size=32)

# Load mô hình đã lưu
loaded_model = load_model('rnn_model.h5')

loss,accuracy_rnn = loaded_model.evaluate( X_test_pad_rnn,y_test_rnn)

print("RNN Accuracy:", accuracy_rnn)



# Routes
@app.route('/')
def index():
    return render_template('index.html',rf_accuracy=rf_accuracy,  rnn_accuracy=accuracy_rnn)

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form.get('model_select')
    input_text = request.form.get('input_text')

    if selected_model == 'random_forest':
        input_text_vectorized = vectorizer_rf.transform([input_text])
        pred_rf = clf_rf.predict(input_text_vectorized)
        if pred_rf == 'REAL':
            result = 'Tin thật (Random Forest)'
        else:
            result = 'Tin giả (Random Forest)'
        return render_template('index.html', result=result ,rf_accuracy=rf_accuracy,  rnn_accuracy=accuracy_rnn)

    elif selected_model == 'rnn':
        input_seq = tokenizer_rnn.texts_to_sequences([input_text])
        input_pad = pad_sequences(input_seq, maxlen=500)
        pred_rnn = loaded_model.predict(input_pad)
        if pred_rnn > 0.5:
            result = 'Tin Thật (RNN)'
        else:
            result = 'Tin Giả (RNN)'
        return render_template('index.html', result=result,rf_accuracy=rf_accuracy,  rnn_accuracy=accuracy_rnn)


if __name__ == '__main__':
    app.run()


