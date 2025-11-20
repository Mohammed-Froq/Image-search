import streamlit as st
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch

# -----------------------------
# Paths
# -----------------------------
LOCAL_MODEL_PATH = "/Users/froquser/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
IMAGE_FOLDER = "/Users/froquser/Desktop/Image_Search/Images"
EMBEDDINGS_FILE = "/Users/froquser/Desktop/Image_Search/clip_image_embeddings_safetensors.pkl"

# -----------------------------
# Device Setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load CLIP model (safetensors)
# -----------------------------
@st.cache_resource
def load_model():
    st.write(f"🧠 Loading CLIP model (safetensors) on {DEVICE.upper()}...")
    model = CLIPModel.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        from_tf=False,                # don't try TensorFlow
        from_flax=False,
        use_safetensors=True          # ✅ force safetensors
    ).to(DEVICE)

    processor = CLIPProcessor.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True
    )
    return model, processor



# -----------------------------
# Compute embeddings for all images
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_embeddings(image_folder):
    model, processor = load_model()
    image_embeddings = {}

    for img_name in tqdm(os.listdir(image_folder), desc="Encoding images"):
        img_path = os.path.join(image_folder, img_name)
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    emb = model.get_image_features(**inputs)
                image_embeddings[img_name] = emb[0].cpu().numpy()
            except Exception as e:
                st.warning(f"⚠️ Skipping {img_name}: {e}")

    # Save embeddings for faster reload
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(image_embeddings, f)

    return image_embeddings


# -----------------------------
# Load cached embeddings if available
# -----------------------------
def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return compute_embeddings(IMAGE_FOLDER)


# -----------------------------
# Find similar images
# -----------------------------
def find_similar(uploaded_image, embeddings, top_k=10):
    model, processor = load_model()
    inputs = processor(images=uploaded_image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        query_emb = model.get_image_features(**inputs)

    query_emb = query_emb[0].cpu().numpy().reshape(1, -1)

    if not embeddings:
        st.error("No embeddings found. Please check your image folder.")
        return []

    names, embs = zip(*embeddings.items())
    sim = cosine_similarity(query_emb, np.vstack(embs))[0]
    top_idx = np.argsort(sim)[::-1][:top_k]
    return [(names[i], float(sim[i])) for i in top_idx]  # return name + similarity



# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CLIP Image Search", page_icon="🔍", layout="wide")
st.title("🔍 Local Image Similarity Search (CLIP Safetensors)")
st.write("Upload an image to find visually similar ones from your local folder using OpenAI’s CLIP model.")

# Check image folder
if not os.path.exists(IMAGE_FOLDER):
    st.error(f"❌ Image folder not found: {IMAGE_FOLDER}")
    st.stop()

st.info("📂 Loading or computing image embeddings...")
embeddings = load_embeddings()
st.success(f"✅ Loaded {len(embeddings)} image embeddings from '{IMAGE_FOLDER}'")

# Upload an image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Finding similar images...")
    similar = find_similar(query_image, embeddings, top_k=10)

    if similar:
        st.subheader("🎯 Most Similar Images (with confidence %):")
        cols = st.columns(3)
        for i, (name, score) in enumerate(similar):
            img_path = os.path.join(IMAGE_FOLDER, name)
            if os.path.exists(img_path):
                confidence_percent = score * 100  # convert 0-1 to 0-100%
                cols[i % 3].image(
                    img_path,
                    caption=f"{name}\nConfidence: {confidence_percent:.1f}%",
                    use_container_width=True
                )
            else:
                cols[i % 3].warning(f"Missing: {name}")
