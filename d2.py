import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Page config
st.set_page_config(
    page_title="Text to Image Generator",
    layout="centered"
)

st.title("🎨 Text to Image Generator")
st.write("Generate images from text using Stable Diffusion")

# Load model (cached to avoid reloading)
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )
    pipe = pipe.to("cpu")  # Streamlit Cloud runs on CPU
    return pipe

pipe = load_model()

# User input
prompt = st.text_input(
    "Enter your prompt:",
    placeholder="A futuristic city at sunset, ultra realistic"
)

# Generate image
if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating image... This may take a while ⏳"):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
