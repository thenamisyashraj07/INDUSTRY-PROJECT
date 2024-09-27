from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Load the trained CNN model
model = tf.keras.models.load_model('models/model.keras')

# Class labels for the species
class_labels = ['species1', 'species2', 'species3', 'species4', 'species5', 
                'species6', 'species7', 'species8', 'species9', 'species10']

# Load endangered species list
with open('endangered_species.txt') as f:
    endangered_species = f.read().splitlines()

# Upload directory for images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to classify the uploaded image
def classify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown", False
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    species_name = class_labels[predicted_class]
    endangered_alert = species_name in endangered_species
    return species_name, endangered_alert

# Function to categorize species and return counts
def get_species_counts_and_lists(species_records):
    endangered_species_list = [s for s, e in species_records if e]
    not_endangered_species_list = [s for s, e in species_records if not e]
    defined_species_list = [s for s, _ in species_records if s != "Unknown"]
    undefined_species_list = [s for s, _ in species_records if s == "Unknown"]

    return {
        'endangered_count': len(endangered_species_list),
        'not_endangered_count': len(not_endangered_species_list),
        'defined_species_count': len(defined_species_list),
        'undefined_species_count': len(undefined_species_list),
        'endangered_species_list': endangered_species_list,
        'not_endangered_species_list': not_endangered_species_list,
        'defined_species_list': defined_species_list,
        'undefined_species_list': undefined_species_list,
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize or retrieve session data
    if 'species_records' not in session:
        session['species_records'] = []

    species_records = session['species_records']
    
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files.get('file')
        if file and file.filename and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            # Classify the uploaded image
            species_name, endangered_alert = classify_image(file_path)
            species_records.append((species_name, endangered_alert))
            session['species_records'] = species_records  # Update session data
            # Get counts and species lists
            species_data = get_species_counts_and_lists(species_records)
            return render_template('index.html', species_name=species_name, endangered_alert=endangered_alert, image_file=file.filename, **species_data)
        else:
            return render_template('index.html', error="No file uploaded or invalid file type.")
    
    # Get counts and species lists for initial view
    species_data = get_species_counts_and_lists(species_records)
    return render_template('index.html', **species_data)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == "__main__":
    app.run(debug=True)
