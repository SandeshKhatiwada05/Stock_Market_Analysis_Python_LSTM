# 📈 Google Stock Price Prediction using LSTM

This project uses **Long Short-Term Memory (LSTM)**, a type of Recurrent Neural Network (RNN), to predict the stock price of **Google** using historical closing price data.

## 🧠 Model Summary

- Model Type: LSTM (Sequential)
- Framework: TensorFlow / Keras
- Data: Historical stock prices (`Google_train_data.csv` and `Google_test_data.csv`)
- Goal: Predict future stock prices based on previous 60 days of closing prices

---

## 📂 Files Included

- `Google_train_data.csv`: Training dataset  
- `Google_test_data.csv`: Test dataset  
- `lstm_stock_predictor.py`: Main Python file with model training and evaluation code  
- `README.md`: This file

---

## 🔧 How It Works

### 🔹 Data Preprocessing
- Load training data (`Close` column)
- Normalize the data using MinMaxScaler (range 0 to 1)
- Create sequences of 60 time steps to predict the 61st value
- Reshape the data into 3D for LSTM input

### 🔹 Model Architecture
![image](https://github.com/user-attachments/assets/0b4cf276-b80c-4f7b-acbe-b41b19a6f517)


```python
model = Sequential()
model.add(Input(shape=(60, 1)))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
```

- **Optimizer**: Adam  
- **Loss Function**: Mean Squared Error

### 🔹 Model Training
- Trained for **200 epochs** with batch size **32**

### 🔹 Evaluation
- Predict on test dataset  
- Inverse the scaled predictions  
- Plot predicted vs. actual Google stock prices

---

## 📊 Sample Output

The model produces a plot showing how closely the predicted values track the actual stock prices over time.
![image](https://github.com/user-attachments/assets/f436b24e-7923-47ed-bb01-1e8ede494ee4)


---

## ▶️ How to Run

### 1. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 2. Place `Google_train_data.csv` and `Google_test_data.csv` in the same directory as the script.

### 3. Run the script:

```bash
python lstm_stock_predictor.py
```

---

## 💡 Future Improvements

- Add volume and other features for multivariate analysis  
- Experiment with different look-back windows  
- Hyperparameter tuning for better accuracy  
- Use GRU or Bidirectional LSTM  

---

## 📌 Author

**Your Name**  
Sandesh Khatiwada
Project for educational purposes (LSTM-based stock prediction)

---

## 📃 License

This project is licensed under the MIT License - feel free to use and modify it for learning purposes.
