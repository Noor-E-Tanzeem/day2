@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe.enable_attention_slicing()
    pipe = pipe.to("cpu")
    return pipe


image = pipe(
    prompt,
    num_inference_steps=15,
    guidance_scale=7.5
).images[0]
