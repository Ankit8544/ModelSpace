from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from diffusers import DiffusionPipeline
import torch
import imageio
import uuid
import os

app = Flask(__name__)

# Load models once
modelscope_model = pipeline(Tasks.text_to_video_synthesis, model='damo-vilab/modelscope-text-to-video-synthesis')
zeroscope_model = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
).to("cuda")

# Output directory
os.makedirs("output", exist_ok=True)

@app.route('/generate-video', methods=['POST'])
def generate_video():
    data = request.get_json()
    
    model_type = data.get("model", "").lower()
    if not model_type:
        return jsonify({"error": "Model is required: 'modelscope' or 'zeroscope'"}), 400

    params = data.get("params", {})
    prompt = params.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required in 'params'"}), 400

    # Common settings
    seed = params.get("seed", 42)
    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("output", output_filename)

    try:
        if model_type == "modelscope":
            # Extract ModelScope-specific parameters
            result = modelscope_model({
                "text": prompt,
                "output_path": output_path,
                "seed": seed,
                "fps": params.get("fps", 8),
                "num_frames": params.get("num_frames", 16),
                "batch_size": params.get("batch_size", 1),
                "decode_audio": params.get("decode_audio", False),
                "resolution": params.get("resolution", "512x512")
            })
            video_path = result["output_path"]

        elif model_type == "zeroscope":
            # Extract Zeroscope-specific parameters
            inference_steps = params.get("num_inference_steps", 25)
            height = params.get("height", 320)
            width = params.get("width", 576)
            fps = params.get("fps", 8)

            generator = torch.manual_seed(seed)

            with torch.autocast("cuda"):
                output = zeroscope_model(
                    prompt,
                    num_inference_steps=inference_steps,
                    height=height,
                    width=width,
                    generator=generator
                )
                video_frames = output.frames

            # Save as video
            imageio.mimsave(output_path, video_frames, fps=fps)
            video_path = output_path

        else:
            return jsonify({"error": f"Unsupported model type: {model_type}"}), 400

        return jsonify({
            "success": True,
            "model": model_type,
            "filename": output_filename,
            "video_path": video_path
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
