import os
import cv2
import uuid
import streamlit as st
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import insightface

# ------------------------
# Config
# ------------------------
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QDRANT_URL = st.secrets.get("QDRANT_URL", "https://e7c5f965-702a-4dbc-8f6a-63fbe338d6b4.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TPnCqWVbbEtGzaZgFRa3GLlkEXMPDpg0dV-0L25A_8Q")
COLLECTION_NAME = "faces_arcface"
EMBED_DIM = 512

# ------------------------
# Qdrant setup (Cloud)
# ------------------------
@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client for Cloud"""
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
        )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant Cloud: {e}")
        raise


def create_collection_if_needed(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = client.get_collections()
        existing_names = [col.name for col in collections.collections]

        if collection_name not in existing_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            st.success(f"Created new collection: {collection_name}")
        
        return True
    except Exception as e:
        st.error(f"Failed to create collection: {e}")
        raise

# ------------------------
# ArcFace model
# ------------------------
@st.cache_resource(show_spinner=False)
def load_arcface_model():
    model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

# ------------------------
# Helpers
# ------------------------
def extract_face_embeddings(model, image):
    faces = model.get(image)
    embeddings = []
    for face in faces:
        embeddings.append((face.embedding, face.bbox))
    return embeddings

def upsert_faces(client, model, image, label: str):
    """Insert labeled face embeddings into Qdrant"""
    faces = model.get(image)
    points = []
    for face in faces:
        embedding = face.embedding.tolist()
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"label": label}
            )
        )
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

def search_face(client, query_embedding, top_k=1):
    """Search for closest embedding in Qdrant"""
    result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k,
    )
    if result:
        return result[0].payload.get("label"), result[0].score
    return "Unknown", 0.0

def annotate_image(model, client, image):
    faces = model.get(image)
    for face in faces:
        label, score = search_face(client, face.embedding)
        bbox = face.bbox.astype(int)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} ({score:.2f})",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return image

def process_video(model, client, video_bytes, output_path):
    nparr = np.frombuffer(video_bytes, np.uint8)
    video = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if video is None:
        st.error("Could not decode video.")
        return None
    
    cap = cv2.VideoCapture(video_bytes)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated = annotate_image(model, client, frame)
        out.write(annotated)

    cap.release()
    out.release()
    return output_path

# ------------------------
# Streamlit UI
# ------------------------
st.title("Face Recognition with ArcFace + Qdrant Cloud")

client = get_qdrant_client()
create_collection_if_needed(client)
model = load_arcface_model()

st.header("1. Register Faces")
uploaded_images = st.file_uploader("Upload reference images", type=["jpg", "png"], accept_multiple_files=True)
label = st.text_input("Label for uploaded faces (e.g., person's name):")

if uploaded_images and label:
    for img_file in uploaded_images:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        n = upsert_faces(client, model, img, label)
        st.success(f"Stored {n} faces for {label}")

st.header("2. Recognize Faces in New Media")
uploaded_media = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4", "avi"])

if uploaded_media:
    file_bytes = uploaded_media.read()
    file_type = uploaded_media.type

    if "image" in file_type:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        annotated = annotate_image(model, client, img)
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{uuid.uuid4()}.jpg")
        cv2.imwrite(out_path, annotated)
        st.image(annotated[:, :, ::-1], caption="Annotated Image")
        st.success(f"Saved result to {out_path}")

    elif "video" in file_type:
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{uuid.uuid4()}.mp4")
        result_path = process_video(model, client, file_bytes, out_path)
        if result_path:
            st.video(result_path)
            st.success(f"Saved result to {result_path}")
