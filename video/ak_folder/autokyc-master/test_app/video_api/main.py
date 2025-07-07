# from fastapi import FastAPI, UploadFile, File
# from cap_self import capture_selfie
# from face_embedding import get_face_embedding
# from face_match import match_faces
# import shutil

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "vKYC API is working"}

# @app.post("/capture-selfie/")
# def capture():
#     filename = capture_selfie()
#     return {"filename": filename}

# @app.post("/upload-and-match/")
# async def upload_and_match(file: UploadFile = File(...)):
#     # Save uploaded image
#     uploaded_path = f"uploaded_{file.filename}"
#     with open(uploaded_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Capture selfie
#     selfie_path = capture_selfie()

#     # Get embeddings
#     selfie_embedding = get_face_embedding(selfie_path)
#     uploaded_embedding = get_face_embedding(uploaded_path)

#     # Match embeddings
#     match, distance = match_faces(selfie_embedding, uploaded_embedding)

#     return {
#         "match": match,
#         "distance": distance,
#         "selfie": selfie_path,
#         "uploaded": uploaded_path
#     }

from fastapi import FastAPI, UploadFile, File
import shutil
from .models.cap_self import capture_selfie
from .models.face_embedding import get_face_embedding
from .models.face_match import is_match
import torch

app = FastAPI()

@app.get("/")
def root():
    return {"message": "vKYC FastAPI is running"}

@app.post("/verify/")
async def verify_face(file: UploadFile = File(...)):
    # Save uploaded file
    uploaded_path = f"uploaded_{file.filename}"
    with open(uploaded_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Capture selfie
    selfie_path = capture_selfie()

    # Get embeddings
    emb1 = get_face_embedding(uploaded_path)
    emb2 = get_face_embedding(selfie_path)

    # Compare
    match, similarity = is_match(emb1, emb2)

    return {
        "match": match,
        "similarity": round(similarity, 4),
        "uploaded_image": uploaded_path,
        "selfie_image": selfie_path
    }
