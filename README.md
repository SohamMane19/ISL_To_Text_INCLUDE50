# **Indian Sign Language Recognition using Deep Learning**

This project is a **Sign Language to Text Recognition System** that uses a deep learning model to classify Indian Sign Language (ISL) signs from video files. It takes `.mov` video files as input, processes them into frames, and predicts the corresponding sign using a trained **LSTM-based model**.

## **Features**
✅ Upload `.mov` video files of hand signs  
✅ Predicts **16 ISL gestures**:  
   - *Howareyou, You, Thankyou, GoodMorning, Extra, Hello, It, He, Loud, Quiet, Alright, Goodafternoon, Happy, Sad, Beautiful, They*  
✅ Uses a pre-trained **LSTM + CNN model** for classification  
✅ FastAPI-based backend for processing and prediction  
✅ Supports **CORS** for frontend integration  
✅ Returns predicted sign label with confidence score  

## **Tech Stack**
- **Backend**: FastAPI, TensorFlow/Keras, OpenCV  
- **Model**: LSTM-based deep learning model  
- **Frontend**: HTML, CSS, JavaScript   
  

## **Installation & Setup**
### **1. Clone the repository**  
```sh
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name
```
### **2. Install dependencies**  
```sh
pip install -r requirements.txt
```
### **3. Run the FastAPI server**  
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```
### **4. Open API documentation**
Visit:  
```
http://127.0.0.1:8000/docs
```
to test the API using Swagger UI.

## **Usage**
1. Upload a `.mov` video file of a sign gesture.  
2. The server processes the video, extracts frames, and predicts the sign.  
3. The predicted **sign label** and **confidence score** are returned as JSON.  

## **API Endpoints**
| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/predict/` | Upload a `.mov` file and get predicted sign |

## **Example Response**
```json
{
  "predicted_sign": "Hello",
  "confidence": 0.92
}
```

## **Future Improvements**
🔹 Add support for more sign gestures  
🔹 Improve model accuracy with more training data  
🔹 Implement a real-time webcam-based sign detection  

---



