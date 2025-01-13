### 1. Imports and class names setup ###
import gradio as gr
import streamlit as st
import os
import torch
from PIL import Image
from model import create_pretrained_vit_model
from timeit import default_timer as timer
from typing import Dict

from model import create_pretrained_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["pizza", "steak", "sushi"]

### 2. Model and transforms preparation ###

# Create EffNetB2 model
pretrained_vit, pretrained_vit_transforms = create_pretrained_vit_model(
    num_classes=3, # len(class_names) would also work
)

# Load saved weights
pretrained_vit.load_state_dict(
    torch.load(
        f="08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth",
        map_location=torch.device("cpu"),  # load to CPU
    ),
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = pretrained_vit_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    pretrained_vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(pretrained_vit(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time




# Set up Streamlit app
st.title("Food Classifier 🍕🥩🍣")
st.write(
    "A Vision Transformer feature extractor computer vision model to classify images of food as pizza, steak, or sushi."
)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Predict button
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction on button click
    if st.button("Run Inference"):
        with st.spinner("Predicting..."):
            predictions, pred_time = predict(image)
        
        # Display predictions
        st.subheader("Predictions:")
        for label, prob in predictions.items():
            st.write(f"{label}: {prob:.2%}")

        # Display prediction time
        st.write(f"Prediction time: {pred_time} seconds")

# ### 4. Gradio app ###

# # Create title, description and article strings
# title = "Food Classifier 🍕🥩🍣"
# description = "A Vision Transformer feature extractor computer vision model to classify images of food as pizza, steak or sushi."
# # article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# # Create examples list from "examples/" directory
# example_list = [["examples/" + example] for example in os.listdir("examples")]

# # Create the Gradio demo
# demo = gr.Interface(fn=predict, # mapping function from input to output
#                     inputs=gr.Image(type="pil"), # what are the inputs?
#                     outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
#                              gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
#                     # Create examples list from "examples/" directory
#                     examples=example_list,
#                     title=title,
#                     description=description,
#                     # article=article
#                     )

# # Launch the demo!
# demo.launch()
