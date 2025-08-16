"""
Streamlit app: Face enrollment + recognition using ArcFace (InsightFace) and Qdrant Cloud

Features
- Enroll faces from uploaded images (incl. HEIF/HEIC) → store 512‑D ArcFace embeddings in **Qdrant Cloud**.
- Recognize faces in uploaded **image or video** (incl. HEVC .mp4/.mov) → draws boxes & names, saves result to `./output`.
- No local vector DB; reads Qdrant Cloud URL/API key from `st.secrets`.

How to run
1) Install requirements:
   pip install streamlit opencv-python-headless numpy pillow pillow-heif imageio[ffmpeg] insightface qdrant-client
2) Put secrets in `.streamlit/secrets.toml`:
   QDRANT_URL = "https://YOUR-INSTANCE.qdrant.tech"
   QDRANT_API_KEY = "your_api_key_here"
3) Run: `streamlit run app.py`

Requirements
- streamlit>=1.36
- opencv-python-headless
- numpy
- pillow
- pillow-heif          # HEIF/HEIC image support
- imageio[ffmpeg]      # HEVC/H.265 video decode
- insightface>=0.7.3
- onnxruntime>=1.17.0  # or onnxruntime-gpu for CUDA
- qdrant-client>=1.7.0
"""

import os
import time
import uuid
import tempfile
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import pillow_heif  # registers HEIF/HEIC support for Pillow
import imageio
import streamlit as st

# Qdrant Cloud
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# InsightFace
from insightface.app import FaceAnalysis

# ------------------------
# Paths & constants
# ------------------------
OUTPUT_DIR = "./output"
UPLOAD_DIR = "./uploads"
COLLECTION_NAME = "faces_arcface"
EMBED_DIM = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

# ------------------------
# Helpers (image/video utils)
# ------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def draw_box(img: np.ndarray, bbox: np.ndarray, name: str, score: float = None) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = name if score is None else f"{name} ({score:.2f})"
    cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def load_image_file(file) -> np.ndarray:
    """Load any supported image (JPEG/PNG/HEIC/HEIF) to BGR numpy array."""
    img = Image.open(file).convert("RGB")
    return pil_to_cv2(img)


def iter_video_frames(path: str):
    """Yield frames (BGR) from video using imageio (supports HEVC/H.265)."""
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps", 25.0)
    except Exception:
        fps = 25.0
    try:
        for frame in reader:
            yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), fps
    finally:
        reader.close()

# ------------------------
# ArcFace model
# ------------------------
@st.cache_resource(show_spinner=False)
def load_face_app(use_gpu: bool = False) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx_id = 0 if use_gpu else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app

# ------------------------
# Qdrant Cloud setup
# ------------------------
QDRANT_URL = st.secrets.get("QDRANT_URL")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY")

@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return client


def ensure_collection(client: QdrantClient, name: str = COLLECTION_NAME):
    cols = client.get_collections().collections
    if name not in [c.name for c in cols]:
        client.create_collection(collection_name=name, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))


def upsert_faces(client: QdrantClient, names: List[str], embeddings: List[np.ndarray]) -> int:
    points = []
    ts = time.time()
    for i, (n, emb) in enumerate(zip(names, embeddings)):
        points.append(
            PointStruct(id=str(uuid.uuid4()), vector=emb.astype(np.float32).tolist(), payload={"name": n, "ts": ts})
        )
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search_face(client: QdrantClient, embedding: np.ndarray, top_k: int = 1):
    res = client.search(collection_name=COLLECTION_NAME, query_vector=embedding.astype(np.float32).tolist(), limit=top_k)
    if not res:
        return None, 0.0
    return res[0].payload.get("name"), float(res[0].score)

# ------------------------
# Face processing
# ------------------------

def detect_and_embed(app: FaceAnalysis, bgr_img: np.ndarray):
    return app.get(bgr_img)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="ArcFace + Qdrant Cloud (HEIF & HEVC)", layout="wide")

st.title("🧠 ArcFace + Qdrant Cloud Face Recognition")
with st.sidebar:
    st.header("Settings")
    use_gpu =False
    threshold = st.slider("Match threshold (cosine)", 0.10, 0.95, 0.50, 0.01)
    frame_stride = st.number_input("Video frame stride", min_value=1, max_value=10, value=2)

# Init services
try:
    client = get_qdrant_client()
    ensure_collection(client)
    st.sidebar.success("✅ Qdrant Cloud connected")
except Exception as e:
    st.error(f"❌ Qdrant Cloud connection failed: {e}")
    st.stop()

face_app = load_face_app(use_gpu)
st.sidebar.success("✅ InsightFace loaded")

st.markdown("---")

enroll_tab, recognize_tab, debug_tab = st.tabs(["🧾 Enroll Faces", "🔎 Recognize (Image/Video)", "🔧 Debug"])

# ------------------------
# Enroll tab
# ------------------------
with enroll_tab:
    st.subheader("Upload images to enroll faces (JPEG/PNG/HEIC/HEIF)")
    enroll_files = st.file_uploader("Select images", type=["jpg", "jpeg", "png", "heic", "heif"], accept_multiple_files=True)

    if enroll_files:
        # Initialize session state for face data if not exists
        if 'face_data' not in st.session_state:
            st.session_state.face_data = []
        
        # Clear previous face data when new files are uploaded
        if 'last_files' not in st.session_state or st.session_state.last_files != [f.name for f in enroll_files]:
            st.session_state.face_data = []
            st.session_state.last_files = [f.name for f in enroll_files]
            
            # Process all uploaded files and detect faces
            for f in enroll_files:
                try:
                    img_bgr = load_image_file(f)
                    faces = detect_and_embed(face_app, img_bgr)
                    st.write(f"**{f.name}** – detected {len(faces)} face(s)")
                    
                    for i, face in enumerate(faces):
                        x1, y1, x2, y2 = map(int, face.bbox)
                        crop = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        
                        # Store face data in session state
                        face_id = f"{f.name}_face_{i}"
                        st.session_state.face_data.append({
                            'id': face_id,
                            'filename': f.name,
                            'face_idx': i,
                            'crop': crop,
                            'embedding': face.embedding.astype(np.float32)
                        })
                        
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")
        
        # Display faces and collect names
        if st.session_state.face_data:
            st.markdown("### Enter names for detected faces:")
            
            # Create columns for better layout
            cols = st.columns(3)
            
            for idx, face_data in enumerate(st.session_state.face_data):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.image(cv2_to_pil(face_data['crop']), 
                            caption=f"Face {face_data['face_idx']+1} from {face_data['filename']}")
                    
                    # Use the face ID as the key to maintain state
                    default_name = f"person_{face_data['id'][-6:]}"  # Use last 6 chars of ID
                    name = st.text_input(
                        f"Name for this face:",
                        value=default_name,
                        key=f"name_{face_data['id']}",
                        placeholder="Enter person's name"
                    )
                    # Store the name in face_data
                    face_data['name'] = name
            
            # Save button
            if st.button("💾 Save all to Qdrant Cloud", type="primary"):
                try:
                    names = [face_data['name'] for face_data in st.session_state.face_data if face_data['name'].strip()]
                    embeddings = [face_data['embedding'] for face_data in st.session_state.face_data if face_data['name'].strip()]
                    
                    if names and embeddings:
                        n = upsert_faces(client, names, embeddings)
                        st.success(f"✅ Saved {n} face embedding(s) to '{COLLECTION_NAME}'.")
                        
                        # Display saved names for confirmation
                        st.info(f"Saved faces: {', '.join(names)}")
                        
                        # Clear the face data after successful save
                        st.session_state.face_data = []
                        if 'last_files' in st.session_state:
                            del st.session_state.last_files
                    else:
                        st.warning("⚠️ Please enter names for at least one face before saving.")
                        
                except Exception as e:
                    st.error(f"❌ Insert failed: {e}")

# ------------------------
# Recognize tab
# ------------------------
with recognize_tab:
    st.subheader("Upload an image or video to recognize faces")
    mode = st.radio("Select input type", ["Image", "Video"], horizontal=True)

    if mode == "Image":
        img_file = st.file_uploader("Upload image (JPEG/PNG/HEIC/HEIF)", type=["jpg", "jpeg", "png", "heic", "heif"], key="rec_image")
        if img_file is not None:
            try:
                bgr = load_image_file(img_file)
                faces = detect_and_embed(face_app, bgr)
                for face in faces:
                    name, score = search_face(client, face.embedding, top_k=1)
                    if name is None or score < threshold:
                        draw_box(bgr, face.bbox, "Unknown")
                    else:
                        draw_box(bgr, face.bbox, name, score)
                out_path = os.path.join(OUTPUT_DIR, f"recognized_{int(time.time())}.jpg")
                cv2.imwrite(out_path, bgr)
                st.image(cv2_to_pil(bgr), caption="Processed image")
                st.success(f"Saved output → {out_path}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    else:  # Video
        vid_file = st.file_uploader("Upload video (MP4/MOV/MKV – HEVC supported)", type=["mp4", "mov", "mkv"], key="rec_video")
        if vid_file is not None:
            # Save to a temp file for imageio
            suffix = os.path.splitext(vid_file.name)[1].lower()
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_in.write(vid_file.read())
            tmp_in.close()

            # Prepare video writer lazily after first frame (to know size/fps)
            out_path = os.path.join(OUTPUT_DIR, f"recognized_{int(time.time())}.mp4")
            writer = None
            frame_idx = 0
            try:
                for frame_bgr, fps in iter_video_frames(tmp_in.name):
                    if frame_idx % int(frame_stride) == 0:
                        faces = detect_and_embed(face_app, frame_bgr)
                        for face in faces:
                            name, score = search_face(client, face.embedding, top_k=1)
                            if name is None or score < threshold:
                                draw_box(frame_bgr, face.bbox, "Unknown")
                            else:
                                draw_box(frame_bgr, face.bbox, name, score)
                    if writer is None:
                        h, w = frame_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(out_path, fourcc, fps or 25.0, (w, h))
                    writer.write(frame_bgr)
                    frame_idx += 1
            except Exception as e:
                st.error(f"Video processing failed: {e}")
            finally:
                if writer is not None:
                    writer.release()
                try:
                    os.remove(tmp_in.name)
                except Exception:
                    pass

            if os.path.exists(out_path):
                st.video(out_path)
                st.success(f"Saved processed video → {out_path}")

# ------------------------
# Debug tab
# ------------------------
with debug_tab:
    st.subheader("Debug Info")
    st.write(f"Qdrant URL: {QDRANT_URL}")
    st.write(f"Collection: {COLLECTION_NAME}")
    st.write(f"Output dir: {os.path.abspath(OUTPUT_DIR)}")
    st.write(f"OpenCV: {cv2.__version__}")
    st.write(f"NumPy: {np.__version__}")
    # Quick check: list up to 10 names
    try:
        pts, _ = get_qdrant_client().scroll(collection_name=COLLECTION_NAME, limit=10)
        names = [p.payload.get("name", "Unknown") for p in pts] if pts else []
        st.write({"stored_names_sample": names})
    except Exception as e:
        st.write({"scroll_error": str(e)})

st.markdown("---")
st.caption(
    "Notes: 1) HEIF/HEIC images supported via pillow-heif. 2) HEVC/H.265 videos decoded via imageio[ffmpeg]. "
    "3) ArcFace embeddings are L2-normalized; Qdrant uses COSINE similarity. Adjust the threshold for your data."
)