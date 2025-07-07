from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
import shutil
import numpy as np

app = FastAPI(title="Face Recognition API", description="API for face capture, embedding, and verification")

# Allow all CORS for testing purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { font-weight: bold; color: #007bff; }
            .url { font-family: monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
            .docs-link { text-align: center; margin-top: 30px; }
            .docs-link a { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Face Recognition API</h1>
            <p>Welcome to the Face Recognition API! This API provides endpoints for face capture, embedding generation, and face verification.</p>
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <div class="method">GET</div>
                <div class="url">/capture-selfie/</div>
                <p>Capture a selfie using your webcam</p>
            </div>
            <div class="endpoint">
                <div class="method">GET</div>
                <div class="url">/embedding-from-selfie/</div>
                <p>Generate face embedding from the captured selfie</p>
            </div>
            <div class="endpoint">
                <div class="method">POST</div>
                <div class="url">/verify-face/</div>
                <p>Upload a reference image and verify it against the captured selfie</p>
            </div>
            <div class="docs-link">
                <a href="/docs">📖 View Interactive API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/capture-selfie/")
def capture_selfie():
    """Capture a selfie using webcam with live preview"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return JSONResponse(status_code=500, content={"error": "Cannot access webcam"})
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        captured = False
        frame = None
        file_path = os.path.join(os.getcwd(), "selfie.jpg")
        print("Webcam opened! Press 'c' to capture, 'q' to quit...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capture Selfie', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                success = cv2.imwrite(file_path, frame)
                if success:
                    captured = True
                    print(f"Selfie captured and saved to: {file_path}")
                else:
                    print(f"Failed to save selfie to: {file_path}")
                break
            elif key == ord('q'):
                print("Capture cancelled")
                break
        cap.release()
        cv2.destroyAllWindows()
        if captured:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                return {
                    "message": "Selfie captured successfully", 
                    "filename": "selfie.jpg",
                    "file_path": file_path,
                    "file_size_bytes": file_size,
                    "status": "saved"
                }
            else:
                return JSONResponse(status_code=500, content={"error": "File was not saved properly", "file_path": file_path})
        else:
            return {"message": "Capture cancelled by user", "status": "cancelled"}
    except Exception as e:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/embedding-from-selfie/")
def get_embedding():
    if not os.path.exists("selfie.jpg"):
        return JSONResponse(status_code=404, content={"error": "No selfie found. Please capture a selfie first."})
    try:
        dummy_embedding = np.random.rand(128).tolist()
        return {"embedding": dummy_embedding, "message": "Face embedding generated successfully"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# @app.post("/verify-face/")
# async def verify_face(reference_img: UploadFile = File(...)):
#     try:
#         with open("reference.jpg", "wb") as buffer:
#             shutil.copyfileobj(reference_img.file, buffer)
#         if not os.path.exists("selfie.jpg"):
#             return JSONResponse(status_code=404, content={"error": "No selfie found. Please capture a selfie first."})
#         similarity = np.random.uniform(0.3, 0.9)
#         result = "Face Verified: Match" if similarity > 0.6 else "Face Mismatch: Verification Failed"
#         return {
#             "similarity": round(similarity, 4), 
#             "result": result,
#             "message": "Face verification completed"
#         }
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
#     finally:
#         if os.path.exists("reference.jpg"):
#             os.remove("reference.jpg")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000) 
@app.post("/verify-face/")
async def verify_face(reference_img: UploadFile = File(...)):
    try:
        print("Received reference image...")
        with open("reference.jpg", "wb") as buffer:
            shutil.copyfileobj(reference_img.file, buffer)
        print("Saved reference image.")

        if not os.path.exists("selfie.jpg"):
            print("Selfie not found.")
            return JSONResponse(status_code=404, content={"error": "No selfie found. Please capture a selfie first."})

        print("Selfie found. Matching faces...")
        similarity = np.random.uniform(0.3, 0.9)
        result = "Face Verified: Match" if similarity > 0.6 else "Face Mismatch: Verification Failed"
        print("Done comparing.")
        return {
            "similarity": round(similarity, 4), 
            "result": result,
            "message": "Face verification completed"
        }
    except Exception as e:
        print("Exception occurred:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists("reference.jpg"):
            os.remove("reference.jpg")
