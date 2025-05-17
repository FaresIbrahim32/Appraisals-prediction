from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'appraisalsecretkey'  # For flash messages

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json'}
MODEL_PATH = 'models'
DATA_PATH = 'data'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for models and data
knn_model = None
scaler = None
potential_comps = None
available_features = ['structure_type', 'gla', 'year_built', 'num_beds', 'distance_km']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load the pre-trained models and data"""
    global knn_model, scaler, potential_comps
    
    try:
        # Try to load KNN model
        if os.path.exists(f"{MODEL_PATH}/property_comp_model.pkl"):
            knn_model = joblib.load(f"{MODEL_PATH}/property_comp_model.pkl")
            logger.info("KNN model loaded successfully")
        else:
            logger.warning("KNN model file not found")
        
        # Try to load scaler
        if os.path.exists(f"{MODEL_PATH}/scaler.pkl"):
            scaler = joblib.load(f"{MODEL_PATH}/scaler.pkl")
            logger.info("Scaler loaded successfully")
        else:
            logger.warning("Scaler file not found")
            
        # Try to load potential comparables
        if os.path.exists(f"{DATA_PATH}/potential_comps.csv"):
            potential_comps = pd.read_csv(f"{DATA_PATH}/potential_comps.csv")
            logger.info(f"Loaded {len(potential_comps)} potential comparable properties")
        else:
            logger.warning("Potential comparables file not found")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

# Load models on startup
load_models()

def bath_to_numeric(bath_str):
    """Convert various bathroom notations to numeric values"""
    if pd.isna(bath_str):
        return 0.0
    
    if isinstance(bath_str, str):
        if ':' in bath_str:
            try:
                full, half = map(float, bath_str.split(':'))
                return full + (half * 0.5)
            except:
                pass
    
    try:
        return float(bath_str)
    except:
        return 0.0

def get_top_comps(subject_property, candidate_properties, knn_model, k=3):
    """Find top k comparable properties"""
    try:
        # Ensure we use a DataFrame with proper column names
        subject_features = subject_property[available_features].copy()
        
        # Find nearest neighbors
        distances, indices = knn_model.kneighbors(subject_features, n_neighbors=min(30, len(candidate_properties)))
        
        # Get the actual properties and their distances
        neighbor_indices = indices[0]
        potential_matches = candidate_properties.iloc[neighbor_indices].copy()
        potential_matches['distance_score'] = distances[0]
        
        # Filter for same structure type first (critical)
        subject_structure = subject_property['structure_type'].values[0]
        structure_matches = potential_matches[potential_matches['structure_type'] == subject_structure].copy()
        
        # If no structure matches, use all potential matches
        if len(structure_matches) < 3:
            structure_matches = potential_matches.copy()
        
        # Consider sale recency if available
        if 'sale_date' in structure_matches.columns:
            mask = ~pd.isna(structure_matches['sale_date'])
            if mask.any():
                current_date = pd.Timestamp.now()
                structure_matches.loc[mask, 'days_since_sale'] = (
                    current_date - pd.to_datetime(structure_matches.loc[mask, 'sale_date'])
                ).dt.days
                
                # Prioritize recent sales
                recent_sales = structure_matches[structure_matches['days_since_sale'] <= 90].copy()
                if len(recent_sales) >= k:
                    structure_matches = recent_sales
        
        # Remove duplicates and get final comps
        structure_matches = structure_matches.drop_duplicates(subset=['address'])
        final_comps = structure_matches.sort_values('distance_score').head(k)
        
        return final_comps
    
    except Exception as e:
        logger.error(f"Error finding comparable properties: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            flash(f'File {filename} uploaded successfully!')
            return redirect(url_for('analyze_file', filename=filename))
    
    return render_template('upload.html')

@app.route('/analyze/<filename>')
def analyze_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Basic file analysis
    file_stats = {
        'filename': filename,
        'size': f"{os.path.getsize(filepath) / 1024:.2f} KB",
        'uploaded_at': os.path.getctime(filepath)
    }
    
    # Try to read the file based on extension
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(filepath)
            file_stats['rows'] = len(df)
            file_stats['columns'] = len(df.columns)
            file_stats['column_names'] = df.columns.tolist()
        except Exception as e:
            flash(f"Error reading CSV: {str(e)}")
    
    elif filename.endswith('.json'):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            file_stats['size_structured'] = f"{len(json.dumps(data)) / 1024:.2f} KB"
        except Exception as e:
            flash(f"Error reading JSON: {str(e)}")
    
    return render_template('analyze.html', stats=file_stats)

@app.route('/find_comps', methods=['GET', 'POST'])
def find_comps():
    if request.method == 'POST':
        try:
            # Check if models are loaded
            if knn_model is None or scaler is None or potential_comps is None:
                flash("Models or data not loaded. Please check server logs.")
                return render_template('find_comps.html')
            
            # Get user input
            input_data = {
                'structure_type': request.form.get('structure_type'),
                'gla': float(request.form.get('gla')),
                'year_built': float(request.form.get('year_built')),
                'num_beds': float(request.form.get('num_beds')),
                'num_baths': request.form.get('num_baths', '0:0')
            }
            
            # If lat/long provided
            lat = request.form.get('latitude')
            lng = request.form.get('longitude')
            
            # Create subject property DataFrame
            subject_property = pd.DataFrame([{
                'structure_type': input_data['structure_type'],
                'gla': input_data['gla'],
                'year_built': input_data['year_built'],
                'num_beds': input_data['num_beds'],
                'distance_km': 0.0  # Will be calculated or set to 0 for self
            }])
            
            # Calculate distances if coordinates provided
            if lat and lng:
                lat = float(lat)
                lng = float(lng)
                
                # If potential_comps has lat/long, calculate distances
                if 'latitude' in potential_comps.columns and 'longitude' in potential_comps.columns:
                    potential_comps['distance_km'] = potential_comps.apply(
                        lambda row: calculate_distance(lat, lng, row['latitude'], row['longitude'])
                        if pd.notna(row['latitude']) and pd.notna(row['longitude'])
                        else 999.0,  # Large value for unknown distances
                        axis=1
                    )
            
            # Scale numeric features
            numeric_features = ['gla', 'year_built', 'num_beds']
            if scaler:
                subject_property[numeric_features] = scaler.transform(subject_property[numeric_features])
            
            # Find top comps
            top_comps = get_top_comps(subject_property, potential_comps, knn_model, k=3)
            
            if top_comps.empty:
                flash("No comparable properties found. Try different criteria.")
                return render_template('find_comps.html')
            
            # Format results for display
            comp_results = []
            for _, comp in top_comps.iterrows():
                # Convert comp to dict for easier template access
                comp_dict = comp.to_dict()
                
                # Add a similarity score
                comp_dict['similarity_score'] = 1.0 / (1.0 + comp_dict.get('distance_score', 0))
                
                # Format any values that need special handling
                if 'sale_price' in comp_dict and pd.notna(comp_dict['sale_price']):
                    comp_dict['sale_price_formatted'] = f"${comp_dict['sale_price']:,.0f}"
                
                comp_results.append(comp_dict)
            
            return render_template('comp_results.html', comps=comp_results, subject=input_data)
            
        except Exception as e:
            logger.error(f"Error in find_comps: {str(e)}")
            flash(f"An error occurred: {str(e)}")
            return render_template('find_comps.html')
    
    # GET request - just show the form
    return render_template('find_comps.html')

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        # This would be the place to trigger model training
        flash("Model training functionality to be implemented.")
        return redirect(url_for('home'))
    
    return render_template('train_model.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)
