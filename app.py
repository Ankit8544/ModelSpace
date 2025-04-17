from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import uuid
import os

app = Flask(__name__)
text2video = pipeline(Tasks.text_to_video_synthesis, model='damo-vilab/modelscope-text-to-video-synthesis')

# Create output directory if not exists
os.makedirs("output", exist_ok=True)

@app.route('/generate-video', methods=['POST'])
def generate_video():
    data = request.get_json()

    # Required
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt (text) is required"}), 400

    # Optional Parameters with defaults
    seed = data.get("seed", 42)
    fps = data.get("fps", 8)
    num_frames = data.get("num_frames", 16)
    batch_size = data.get("batch_size", 1)
    decode_audio = data.get("decode_audio", False)  # Usually False
    resolution = data.get("resolution", "512x512")  # If supported (try "256x256", "512x512")

    # Output path
    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = f"output/{output_filename}"

    try:
        result = text2video({
            "text": prompt,
            "output_path": output_path,
            "seed": seed,
            "fps": fps,
            "num_frames": num_frames,
            "batch_size": batch_size,
            "decode_audio": decode_audio,
            "resolution": resolution
        })

        return jsonify({
            "success": True,
            "video_path": result["output_path"],
            "filename": output_filename
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

