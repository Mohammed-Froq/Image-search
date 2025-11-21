# 🔍 CLIP Image Search (Safetensors)

A local image similarity search app using OpenAI’s **CLIP model (ViT-B/32)** with safetensors.  
Upload an image and find visually similar images from a local image folder using embeddings computed by CLIP.

---

## Packshot Images (Testing Purpose)

- Use this Folder for testing Purpose [Images Folder](https://drive.google.com/file/d/1qCX9t3g9tYqD_3Qv1PSjiKjLoGwjsYFT/view?usp=sharing)

---

## 📑 Table of Contents

1. [Features](#features)  
2. [Demo](#demo)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Folder Structure](#folder-structure)  
6. [How it Works (Technical Details)](#how-it-works-technical-details)  
7. [Model Information](#model-information)  
8. [Usage](#usage)  
9. [License](#license)  

---

## **Features**

- 🔹 Compute embeddings for all images in a local folder  
- 🔹 Find the top-K visually similar images to a query image  
- 🔹 Efficient caching with `pickle` to avoid recomputing embeddings  
- 🔹 GPU support using PyTorch (`cuda`) if available  
- 🔹 Streamlit UI for interactive use  

---

## **Demo**

- Upload an image (JPEG/PNG)  
- View **top 10 similar images** with confidence %  
- Works entirely offline using local CLIP model  

---

## **Requirements**

Python 3.10+ and the following libraries:

streamlit==1.29.0
numpy==1.26.0
Pillow==10.0.0
tqdm==4.66.1
torch==2.2.0
transformers==4.54.0
scikit-learn==1.3.1
safetensors==0.3.2

---

## **Installation**

**1️⃣ Clone the repo**
git clone https://github.com/Mohammed-Froq/Image-search.git
cd Image-search

**2️⃣ Create Python environment**
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

**3️⃣ Install dependencies**
pip install -r requirements.txt

**4️⃣ Download the CLIP ViT-B/32 model (safetensors)**

- HuggingFace model: openai/clip-vit-base-patch32
- Save locally in LOCAL_MODEL_PATH:

~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/<snapshot_hash>

---

## **Folder Structure**

Image-search/
|
+-- main.py                                # Main Streamlit app
+-- clip_image_embeddings_safetensors.pkl  # Cached embeddings
+-- Images/                                # Folder containing images to search
+-- requirements.txt                       # Python dependencies

---

## **How it Works (Technical Details)**

**1️⃣ Model Loading**
- Uses CLIP ViT-B/32 in safetensors format:

model = CLIPModel.from_pretrained(LOCAL_MODEL_PATH, use_safetensors=True)
processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_PATH)

**2️⃣ Compute Image Embeddings**

- Convert each image to RGB
- Compute embeddings (512-dimensional feature vectors)
- Save embeddings in clip_image_embeddings_safetensors.pkl for caching

emb = model.get_image_features(**processor(images=image, return_tensors="pt").to(DEVICE))

**3️⃣ Similarity Search**

- Compute embedding for query image
- Compute cosine similarity with cached embeddings:

sim = cosine_similarity(query_emb, np.vstack(all_embeddings))

- Returns top-K matches with similarity scores (0–1)

**4️⃣ Streamlit Interface**

- Upload an image
- View top-K similar images in columns with confidence %

---

## **Model Information**

1.Model Name: CLIP ViT-B/32
2.Provider: OpenAI
3.HuggingFace Link: clip-vit-base-patch32
4.Type: Vision Transformer (ViT)
5.Input Size: 224x224 RGB images
6.Output: 512-dimensional feature vectors
7.Weights Format: safetensors (safe and fast for PyTorch)

- Using local files ensures offline functionality and avoids repeated downloads.

---

## **Usage**

- Make sure images are in the Images/ folder

**Run the app:**

- streamlit run main.py

**Open your browser at http://localhost:8501**

- Upload an image and see top similar images with confidence scores

---

## **License**

MIT License — Free to use for personal and commercial projects
