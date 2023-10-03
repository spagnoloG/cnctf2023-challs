from flask import (
    Flask,
    render_template,
    session,
    request,
    redirect,
    url_for,
    send_from_directory,
)
import os
import random
import xml.etree.ElementTree as ET

app = Flask(__name__)

app.secret_key = "s3cr3t_c0d3_4pp_133t"
BASE_PATH = "dataset_val/"
IMAGE_FILES = sorted([f for f in os.listdir(BASE_PATH) if f.endswith(".JPEG")])
XML_FILES = sorted([f for f in os.listdir(BASE_PATH) if f.endswith(".xml")])
FLAG = os.environ.get("FLAG", "flag{this_is_a_fake_flag}")


@app.route("/")
def index():
    if "input_count" not in session:
        session["input_count"] = 0
        session["correct_ones"] = 0
        session["wrong_ones"] = 0

    if session["input_count"] >= 100:
        session.pop("input_count", None)
        if session["correct_ones"] >= 70:
            return FLAG
        return "You've reached the limit of 100 inputs!"

    random_image = random.choice(IMAGE_FILES)
    session["current_image"] = random_image

    return render_template("index.html", image=random_image)


@app.route("/submit", methods=["POST"])
def submit():
    session["input_count"] = session.get("input_count", 0) + 1
    user_input = request.form["user_input"]
    filename = session.get("current_image", None)
    if filename is None:
        return "No image selected!"

    # Open xml file
    tree = ET.parse(BASE_PATH + filename[:-5] + ".xml")
    root = tree.getroot()
    for obj in root.iter("object"):
        for name in obj.iter("name"):
            ground_truth = name.text

    correct_ones = session.get("correct_ones", 0)
    wrong_ones = session.get("wrong_ones", 0)
    if user_input == ground_truth:
        correct_ones += 1
        session["correct_ones"] = correct_ones
    else:
        wrong_ones += 1
        session["wrong_ones"] = wrong_ones

    return redirect(url_for("index"))


@app.route("/image/<filename>")
def send_image(filename):
    return send_from_directory(BASE_PATH, filename)


@app.route("/xml/<filename>")
def send_xml(filename):
    return send_from_directory(BASE_PATH, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
