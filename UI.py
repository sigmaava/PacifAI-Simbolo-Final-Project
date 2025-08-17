import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
import pickle
from tensorflow.keras.models import load_model # type: ignore
import joblib

# Set page title and layout
st.set_page_config(page_title="AI for Peace", layout="centered")

# Title of the app
st.title("AI for Peace")
st.write("Welcome! This app analyzes text and images for violence/non-violence, hate speech, topic classification, and real/fake news.")

# User choice: Text or Image
input_type = st.radio("Choose input type:", ("Text", "Image"))

# Initialize a variable to store input data
input_data = None


# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

#Loading models
with open('models/fakeNewsDetection.pkl','rb') as f:
   FakeNewsModel = pickle.load(f)

with open('models/TopicAnalysisModel.pkl','rb') as f:
   TopicModel = pickle.load(f)

@st.cache_resource

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)
    return denoised

def load_violence_model():
    model = load_model('models/violence_detection_model.keras')
    return model

ViolenceModel = load_violence_model()

vectorizer = joblib.load('models/vectorizer_for_hate_speech.pkl')
hateSpeechModel = joblib.load('models/multinomialNB_for_hate_speech.pkl')
    
def showTextResults(data):
    input_data = data["input"]
    isHate = data["isHate"]
    topic = data["topicOfText"]
    isFake = data["realOrFake"]

    st.write("Analysis Results:")
    st.subheader(f"Input Text: {input_data}")
    st.subheader(f"Hate Speech: :{"red" if isHate == "Hate" else "green"}[{isHate}]")
    st.subheader(f"Topic of the Text: {topic}")
    st.subheader(f"Real/Fake News: :{"red" if isFake == "Fake" else "green"}[{isFake}]")

def analyseText(text):
    realOrFake = FakeNewsModel.predict([text])[0]
    # preparing outputs
    realOrFake = "Real" if realOrFake == "true" else "Fake"

    topicOfText = TopicModel.predict([input_data])[0]

    isHate = hateSpeechModel.predict(vectorizer.transform([text]))[0]

    predictedRes = {
        "input": text,
        "realOrFake": realOrFake,
        "topicOfText": topicOfText,
        "isHate": isHate
    }
    return predictedRes

def process_image(uploaded_image):
    # Convert the uploaded image to a format suitable for OpenCV
    img = np.array(uploaded_image)

    # ----- Violence Detection -----
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = ViolenceModel.predict(img_array)
    violence_detected = "Violent" if prediction[0][0] > 0.5 else "Non-Violent"

    # ----- OCR Processing -----
    preprocessed_img = preprocess_for_ocr(img)
    extracted_text = pytesseract.image_to_string(preprocessed_img)

    return violence_detected, extracted_text

if input_type == "Text":
    # Text input box
    user_text = st.text_area("Enter your text here:")
    if st.button("Analyze Text"):
        if user_text.strip():  # Check if text is not empty
            input_data = user_text
            st.success("Text received! Processing...")
            # passing to pre-trained text analysis models
            
            # Displaying results 
            showTextResults(analyseText(input_data))
        else:
            st.warning("Please enter some text to analyze.")

elif input_type == "Image":
    # Image uploader
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image",width=350)

        image_display = Image.open(uploaded_image)
        
        with st.spinner('Processing...'):
            violence_result, extracted_text = process_image(image_display)

        # Classify the image as violence/non-violence
        st.markdown("🔒 Violence Detection Result:")
        st.subheader(f"**{violence_result}**")     
        
        # displaying extracted text
        if extracted_text.strip():  # Check if text is detected
            st.success("Text extracted from the image!")
            st.write(f"Extracted Text: {extracted_text}")
            input_data = extracted_text
            
            showTextResults(analyseText(input_data))
        else:
            st.warning("No text detected in the image.")

# Footer
st.markdown("---")
st.write("Built with ❤️ for peace and harmony.")