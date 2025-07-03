import io
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from model_utils import load_model, predict_image, load_label_binarizer
from transforms import preprocess_image
from cam_utils import cam_img, denormalize

st.set_page_config(
    page_title="Chest X-ray Diagnosis",
    #layout="wide"
)

st.markdown(
    """
    <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #333;
        background-color: #fafafa;
    }
    .block-container {
        padding: 2rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_artifacts():
    mlb = load_label_binarizer("models/mlb_classes.pkl")
    model = load_model("models/best_model.pth", num_classes=len(mlb.classes_))
    return model, mlb

st.title("Chest X-ray Diagnosis")
st.caption("Automated classification with Grad-CAM visual explanation.")

with st.spinner("Loading model..."): #spinner around the first call
    model, mlb = load_artifacts()
label_names = mlb.classes_


SAMPLE_IMAGE_DIR = "app/assets/sample_images"
sample_images = sorted([f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.lower().endswith(("png", "jpg", "jpeg"))])

option = st.radio(
    "Choose an input method:",
    ("Upload your own image", "Use a sample image"),
    horizontal=True
)

image = None
filename = None


if option == "Upload your own image":
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        filename = uploaded_file.name
        st.image(image, caption="Uploaded X-ray")

elif option == "Use a sample image":
    selected_sample = st.selectbox("Choose a sample image", sample_images)
    image_path = os.path.join(SAMPLE_IMAGE_DIR, selected_sample)
    image = Image.open(image_path).convert("RGB")
    filename = selected_sample
    st.image(image, caption=f"Sample Image: {selected_sample}")


if image is not None:
    run_button = st.button("Run Analysis")

    if run_button:

        image_tensor = preprocess_image(image)
        predicted_labels, pred_proba = predict_image(model,image_tensor)
        decoded_labels = mlb.inverse_transform(predicted_labels.cpu().numpy())

        progress_bar = st.progress(value=0, text="Analyzing image...")

        rows, cols = 3, 5
        step = 100 / (rows*cols)

        fig = plt.figure(figsize=(16,9))
        for i in range(rows * cols):
            plt.subplot(rows, cols, i + 1)

            if i == 0:
                img_denorm = denormalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img_np = img_denorm.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)

                plt.imshow(img_np)
                plt.axis('off')
                plt.title(", ".join(decoded_labels[0]))

            else:

                is_pred = predicted_labels[0, i - 1].item() == 1
                prob = float(pred_proba[i - 1])

                img, target_class = cam_img(model, image_tensor, i - 1)
                plt.imshow(img)
                plt.axis("off")
                title = f"{label_names[target_class]}; {prob * 100:.2f}%"
                color = "darkgreen" if is_pred else "black"
                plt.title(f"Grad-CAM for: {title}", color=color)

            progress_bar.progress(value=int((i+1)*step), text="Analyzing image...")

        plt.tight_layout()

        # save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.session_state.gradcam_buffer = buf
        st.session_state.last_filename = filename

# show results
if "gradcam_buffer" in st.session_state:
    st.divider()
    st.subheader("Model Visual Explanation (Grad-CAM)")
    st.image(st.session_state.gradcam_buffer)

    st.download_button(
        label="Download Image",
        data=st.session_state.gradcam_buffer,
        file_name="gradcam_results.png",
        mime="image/png"
    )




