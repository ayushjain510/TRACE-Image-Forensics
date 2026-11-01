"""Flask web app: upload an image, get a tampering verdict with ELA,
a suspicion heatmap and copy-move matches.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import base64
import io
import os
import sys
import uuid

from flask import Flask, redirect, render_template, request, url_for

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import VALID_EXT  # noqa: E402
from predict import analyse  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def to_data_uri(pil_img, max_side=640):
    img = pil_img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse_route():
    file = request.files.get("image")
    if not file or not file.filename:
        return render_template("index.html", error="Please choose an image.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in VALID_EXT:
        return render_template("index.html",
                               error=f"Unsupported file type '{ext}'.")

    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(path)
    try:
        from PIL import Image
        original = Image.open(path).convert("RGB")
        r = analyse(path)
    except FileNotFoundError as exc:
        return render_template("index.html", error=str(exc))
    except Exception as exc:
        return render_template("index.html", error=f"Could not analyse image: {exc}")
    finally:
        if os.path.exists(path):
            os.remove(path)

    return render_template(
        "index.html",
        result=r,
        filename=file.filename,
        original=to_data_uri(original),
        ela=to_data_uri(r["ela_image"]),
        heatmap=to_data_uri(r["heatmap_image"]),
        copymove=to_data_uri(r["copy_move_image"]),
    )


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="File is larger than 16 MB."), 413


if __name__ == "__main__":
    app.run(debug=False)
