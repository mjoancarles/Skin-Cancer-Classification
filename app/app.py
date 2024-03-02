import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import Model

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model')  # Update the model path
    return model

# Based on the Keras offitial website: https://keras.io/examples/vision/grad_cam/
def grad_cam(model, image, class_idx, layer_name='conv2d_3'):
    # Get the last convolutional layer
    last_conv_layer = model.get_layer(layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    # Get the classifier part of the model
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = Model(classifier_input, x)

    # Get the gradient of the specified class with respect to the output feature map of last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(np.array([image]))
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, class_idx]

    # This is the gradient of the specified class with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by 'how important this channel is' with regard to the specified class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

# Overlay the heatmap on original image
def overlay_heatmap(heatmap, original_image, alpha=0.4):  # Adjusted alpha to 0.4 as per your latest code
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + original_image * (1 - alpha)
    
    # Normalize the image
    superimposed_img = superimposed_img - superimposed_img.min()  # Shift the minimum value to 0.0
    superimposed_img = superimposed_img / superimposed_img.max()  # Scale the maximum value to 1.0

    return superimposed_img

model = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Home", "About Melanomas"])

if choice == "Home":
    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    st.title('Skin Lesions Detection App')
    
    st.write("""
        Melanoma is a serious form of skin cancer that originates in the pigment-producing melanocytes. 
        Early detection and treatment are crucial for a positive outcome. If you're interested in learning 
        more about melanomas, their characteristics, and their treatment, please visit the 'About Melanomas' tab.
        
        This app pretends to classify skin lesions into 7 different categories in which melanoma is one of them.
        When uploading a picture, if the model predicts it as a melanoma, please consult a dermatologist.
    """)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    def preprocess_and_predict(image, model, target_size=(225, 300)):
        if image is not None:
            # Preprocess the image
            img = Image.open(image)
            img = img.resize(target_size)  # Adjust the size to your model's input size
            img_array = np.transpose(img, (1, 0, 2))  # Transpose to (225, 300, 3) to match model's input shape
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(predictions)]
            return predicted_class, predictions, img_array[0]  # Return the preprocessed image too

    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
    
        with col2:  # Using the middle column to display the image
            st.image(uploaded_file, caption='Uploaded Image', width=300)

        # Display a loading spinner while classifying
        with st.spinner("Classifying..."):
            predicted_class, probabilities, preprocessed_image = preprocess_and_predict(uploaded_file, model)
        
        # Display the prediction
        st.write(f"The lesion is likely to be {predicted_class} with confidence:")
        # Convert the probabilities to a DataFrame for nicer display
        probabilities_df = pd.DataFrame({
            'Class': class_names,
            'Probability': probabilities
        })
        
        # Display the probabilities as a table
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.table(probabilities_df.style.format({'Probability': "{:.2f}"}))
        
        # Explanation text for the heatmap
        st.write("The highlighted areas in the following image show where the model focused its attention while making the prediction.")

        # Generate heatmap for the predicted class
        heatmap = grad_cam(model, preprocessed_image, class_names.index(predicted_class))

        # Overlay heatmap on the original image
        superimposed_img = overlay_heatmap(heatmap, preprocessed_image, alpha=0.4)  # alpha can be adjusted as needed

        # Display the heatmap
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(superimposed_img, caption='Grad-CAM Heatmap Overlay', width=300)

elif choice == "About Melanomas":
    st.title("About Melanomas")
    st.write("""
    Skin cancer can occur anywhere on the body, but it is most common in skin that is often exposed to sunlight, such as the face, neck, hands, and arms. There are different types of cancer that start in the skin.

    Melanoma is a disease in which malignant (cancer) cells form in the skin cells called melanocytes (cells that color the skin). Melanocytes are found throughout the lower part of the epidermis. They make melanin, the pigment that gives skin its natural color.
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('melanoma.jpg', caption='Melanoma Anatomy')
    
    st.write("""
    Here are some key points of the disease:
    
    Rate of New Cases and Deaths per 100,000: The rate of new cases of melanoma of the skin was 21.0 per 100,000 men and women per year. The death rate was 2.1 per 100,000 men and women per year. These rates are age-adjusted and based on 2016–2020 cases and deaths.

    Lifetime Risk of Developing Cancer: Approximately 2.2 percent of men and women will be diagnosed with melanoma of the skin at some point during their lifetime, based on 2017-2019 data.

    Prevalence of This Cancer: In 2020, there were an estimated 1,413,976 people living with melanoma of the skin in the United States.
    
    SOURCE: https://seer.cancer.gov/statfacts/html/melan.html
    """)
