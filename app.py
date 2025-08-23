"""
Streamlit app: Face enrollment + recognition using ArcFace (ONNX) and Qdrant Cloud

Replaces InsightFace with:
- OpenCV DNN face detector (Res10 SSD).
- ArcFace ResNet100 (ONNX, 512-D embeddings) via onnxruntime.

Models auto-download to ./models from stable sources:
- ArcFace: OpenVINO OMZ hosting (primary) with GitHub-raw fallback.
- OpenCV face detector: official OpenCV repos.

Features kept:
- Enroll faces from uploaded images (incl. HEIF/HEIC) → store 512-D embeddings in Qdrant Cloud.
- Recognize faces in uploaded image or video (incl. HEVC .mp4/.mov) → draw boxes & names, save to ./output.
- No local vector DB; reads Qdrant Cloud URL/API key from st.secrets.
"""

import os
import time
import uuid
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import pillow_heif  # registers HEIF/HEIC support for Pillow
import imageio
import streamlit as st

# Qdrant Cloud
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ONNX Runtime
try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError(
        "onnxruntime is required. Install with:\n"
        "pip install --no-cache-dir onnxruntime  # or onnxruntime-gpu"
    )

import onnx
from onnx import helper

# ------------------------
# Paths & constants
# ------------------------
OUTPUT_DIR = "./output"
UPLOAD_DIR = "./uploads"
MODEL_DIR = "./models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COLLECTION_NAME = "faces_arcface"
EMBED_DIM = 512

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

# Stable model URLs (primary + fallback)
ARC_URLS = [
    # Primary: OpenVINO OMZ hosting (stable direct link)
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/face-recognition-resnet100-arcface-onnx/arcfaceresnet100-8.onnx",
    # Fallback: ONNX model zoo raw
    "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
]
PROTO_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
]
CAFFE_URLS = [
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
]

ARC_ONNX = os.path.join(MODEL_DIR, "arcfaceresnet100-8.onnx")
FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_CAFFE = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


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
# Downloads
# ------------------------
def _download_first_ok(urls: List[str], dst: str, min_bytes: int = 1024 * 100) -> None:
    """Try URLs in order until one works; rudimentary size check."""
    import urllib.request, shutil

    if os.path.exists(dst) and os.path.getsize(dst) >= min_bytes:
        return

    last_err = None
    for url in urls:
        try:
            tmp = dst + ".tmp"
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.getsize(tmp) < min_bytes:
                os.remove(tmp)
                continue
            os.replace(tmp, dst)
            return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download to {dst}. Last error: {last_err}")


# ------------------------
# ONNX model fixer
# ------------------------
def fix_batchnorm_spatial(model_path: str):
    """Patch BatchNormalization nodes to force spatial=1 (ArcFace ONNX bug)."""
    try:
        model = onnx.load(model_path)
        changed = False
        for node in model.graph.node:
            if node.op_type == "BatchNormalization":
                has_spatial = any(attr.name == "spatial" for attr in node.attribute)
                if not has_spatial:
                    node.attribute.append(helper.make_attribute("spatial", 1))
                    changed = True
                else:
                    for attr in node.attribute:
                        if attr.name == "spatial" and attr.i != 1:
                            attr.i = 1
                            changed = True
        if changed:
            onnx.save(model, model_path)
            print(f"Patched BatchNormalization nodes in {model_path}")
    except Exception as e:
        print(f"Failed to patch ONNX model {model_path}: {e}")


# ------------------------
# Models (face detector + ArcFace)
# ------------------------
@st.cache_resource(show_spinner=False)
def load_face_detector() -> cv2.dnn_Net:
    _download_first_ok(PROTO_URLS, FACE_PROTO, min_bytes=4_000)  # prototxt small
    _download_first_ok(CAFFE_URLS, FACE_CAFFE, min_bytes=5_000_000)
    net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_CAFFE)
    return net


@st.cache_resource(show_spinner=False)
def load_arcface_session(use_gpu: bool = False) -> Tuple[ort.InferenceSession, str, str]:
    _download_first_ok(ARC_URLS, ARC_ONNX, min_bytes=20_000_000)
    # Patch BatchNormalization bug before loading
    fix_batchnorm_spatial(ARC_ONNX)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(ARC_ONNX, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name



def detect_faces_dnn(net: cv2.dnn_Net, bgr: np.ndarray, conf_thresh: float = 0.5) -> List[np.ndarray]:
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= conf_thresh:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append(np.array([x1, y1, x2, y2], dtype=int))
    return boxes


def _crop_align(bgr: np.ndarray, box: np.ndarray, margin: float = 0.2, size: int = 112) -> np.ndarray:
    """Crop with a bit of margin and resize to ArcFace input size. (No landmarks alignment.)"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    m = max(w, h) * (1 + margin)
    nx1 = int(max(0, cx - m / 2))
    ny1 = int(max(0, cy - m / 2))
    nx2 = int(min(bgr.shape[1] - 1, cx + m / 2))
    ny2 = int(min(bgr.shape[0] - 1, cy + m / 2))
    crop = bgr[ny1:ny2, nx1:nx2].copy()
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def embed_arcface(sess: ort.InferenceSession, in_name: str, out_name: str, bgr_face112: np.ndarray) -> np.ndarray:
    """Returns L2-normalized 512-D embedding."""
    rgb = cv2.cvtColor(bgr_face112, cv2.COLOR_BGR2RGB).astype(np.float32)
    # ArcFace preprocessing: 1x3x112x112, (x-127.5)/128.0
    chw = np.transpose(rgb, (2, 0, 1))
    chw = (chw - 127.5) / 128.0
    inp = np.expand_dims(chw, axis=0).astype(np.float32)
    emb = sess.run([out_name], {in_name: inp})[0].reshape(-1)
    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def detect_and_embed(det_net: cv2.dnn_Net,
                     sess: ort.InferenceSession,
                     in_name: str,
                     out_name: str,
                     bgr_img: np.ndarray,
                     conf_thresh: float = 0.5):
    """Mimic InsightFace .get(): returns list-like of objects with .bbox and .embedding"""
    class _Face:
        __slots__ = ("bbox", "embedding")

        def __init__(self, bbox, embedding):
            self.bbox = bbox
            self.embedding = embedding

    faces = []
    boxes = detect_faces_dnn(det_net, bgr_img, conf_thresh=conf_thresh)
    for box in boxes:
        face_crop = _crop_align(bgr_img, box, margin=0.2, size=112)
        if face_crop is None:
            continue
        emb = embed_arcface(sess, in_name, out_name, face_crop)
        faces.append(_Face(box, emb))
    return faces


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
    for n, emb in zip(names, embeddings):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=emb.astype(np.float32).tolist(), payload={"name": n, "ts": ts}))
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search_face(client: QdrantClient, embedding: np.ndarray, top_k: int = 1):
    res = client.search(collection_name=COLLECTION_NAME, query_vector=embedding.astype(np.float32).tolist(), limit=top_k)
    if not res:
        return None, 0.0
    return res[0].payload.get("name"), float(res[0].score)


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="ArcFace (ONNX) + Qdrant Cloud (HEIF & HEVC)", layout="wide")

st.title("🧠 ArcFace (ONNX) + Qdrant Cloud Face Recognition")
with st.sidebar:
    st.header("Settings")
    use_gpu = False
    threshold = st.slider("Match threshold (cosine)", 0.10, 0.95, 0.50, 0.01)
    frame_stride = st.number_input("Video frame stride", min_value=1, max_value=10, value=2)
    det_conf = st.slider("Detector confidence", 0.1, 0.99, 0.5, 0.01)

# Init services
try:
    client = get_qdrant_client()
    ensure_collection(client)
    st.sidebar.success("✅ Qdrant Cloud connected")
except Exception as e:
    st.error(f"❌ Qdrant Cloud connection failed: {e}")
    st.stop()

# Load models
try:
    det_net = load_face_detector()
    sess, in_name, out_name = load_arcface_session(use_gpu)
    st.sidebar.success("✅ Models loaded (OpenCV DNN + ArcFace ONNX)")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

st.markdown("---")

enroll_tab, recognize_tab, debug_tab = st.tabs(["🧾 Enroll Faces", "🔎 Recognize (Image/Video)", "🔧 Debug"])

# ------------------------
# Enroll tab
# ------------------------
with enroll_tab:
    st.subheader("Upload images to enroll faces (JPEG/PNG/HEIC/HEIF)")
    enroll_files = st.file_uploader("Select images", type=["jpg", "jpeg", "png", "heic", "heif"], accept_multiple_files=True)

    if enroll_files:
        if 'face_data' not in st.session_state:
            st.session_state.face_data = []

        if 'last_files' not in st.session_state or st.session_state.last_files != [f.name for f in enroll_files]:
            st.session_state.face_data = []
            st.session_state.last_files = [f.name for f in enroll_files]

            for f in enroll_files:
                try:
                    img_bgr = load_image_file(f)
                    faces = detect_and_embed(det_net, sess, in_name, out_name, img_bgr, conf_thresh=det_conf)
                    st.write(f"**{f.name}** – detected {len(faces)} face(s)")
                    for i, face in enumerate(faces):
                        x1, y1, x2, y2 = map(int, face.bbox)
                        crop = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
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

        if st.session_state.face_data:
            st.markdown("### Enter names for detected faces:")
            cols = st.columns(3)
            for idx, face_data in enumerate(st.session_state.face_data):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.image(cv2_to_pil(face_data['crop']),
                             caption=f"Face {face_data['face_idx']+1} from {face_data['filename']}")
                    default_name = f"person_{face_data['id'][-6:]}"
                    name = st.text_input("Name for this face:",
                                         value=default_name,
                                         key=f"name_{face_data['id']}",
                                         placeholder="Enter person's name")
                    face_data['name'] = name

            if st.button("💾 Save all to Qdrant Cloud", type="primary"):
                try:
                    names = [d['name'] for d in st.session_state.face_data if d['name'].strip()]
                    embeddings = [d['embedding'] for d in st.session_state.face_data if d['name'].strip()]
                    if names and embeddings:
                        n = upsert_faces(client, names, embeddings)
                        st.success(f"✅ Saved {n} face embedding(s) to '{COLLECTION_NAME}'.")
                        st.info(f"Saved faces: {', '.join(names)}")
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
                faces = detect_and_embed(det_net, sess, in_name, out_name, bgr, conf_thresh=det_conf)
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
            suffix = os.path.splitext(vid_file.name)[1].lower()
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_in.write(vid_file.read())
            tmp_in.close()

            out_path = os.path.join(OUTPUT_DIR, f"recognized_{int(time.time())}.mp4")
            writer = None
            frame_idx = 0
            try:
                for frame_bgr, fps in iter_video_frames(tmp_in.name):
                    if frame_idx % int(frame_stride) == 0:
                        faces = detect_and_embed(det_net, sess, in_name, out_name, frame_bgr, conf_thresh=det_conf)
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
    try:
        pts, _ = get_qdrant_client().scroll(collection_name=COLLECTION_NAME, limit=10)
        names = [p.payload.get("name", "Unknown") for p in pts] if pts else []
        st.write({"stored_names_sample": names})
    except Exception as e:
        st.write({"scroll_error": str(e)})

st.markdown("---")
st.caption(
    "Notes: 1) HEIF/HEIC images supported via pillow-heif. 2) HEVC/H.265 videos decoded via imageio[ffmpeg]. "
    "3) ArcFace ONNX generates 512-D embeddings; Qdrant uses COSINE similarity. You can tune the threshold for your data."
)
