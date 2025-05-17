import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Set page configuration
st.set_page_config(
    page_title="Property Comparison System",
    page_icon="🏠",
    layout="wide"
)

# Load models and data
@st.cache_resource  # This caches the loaded models
def load_models():
    try:
        # Check if models exist locally
        if os.path.exists("model/property_comp_model.pkl") and os.path.exists("model/scaler.pkl"):
            knn_model = joblib.load("model/property_comp_model.pkl")
            scaler = joblib.load("model/property_comp_model.pkl")
            potential_comps = pd.read_csv("data/potential_comps.csv")
            return knn_model, scaler, potential_comps
        else:
            st.error("Model files not found. Please make sure models are properly saved.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Function to convert bathrooms to numeric
def bath_to_numeric(bath_str):
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

# Function to find comparable properties
def get_top_comps(subject_property, candidate_properties, knn_model, available_features, k=3):
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
        
        # Filter for same structure type first
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
        st.error(f"Error finding comparable properties: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Main app
def main():
    # App title
    st.title("Property Comparison System")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "Find Comparable Properties", "About"])
    
    # Load models
    knn_model, scaler, potential_comps = load_models()
    available_features = ['structure_type', 'gla', 'year_built', 'num_beds', 'distance_km']
    
    # Home page
    if page == "Home":
        st.write("## Welcome to the Property Comparison System")
        st.write("This application helps you find comparable properties based on key features.")
        
        st.write("### Features:")
        st.write("- Find top 3 comparable properties based on your property details")
        st.write("- Compare property features side by side")
        st.write("- View similarity scores and explanations")
        
        st.info("Click on 'Find Comparable Properties' in the sidebar to get started!")
        
        # Display sample data if available
        if potential_comps is not None:
            st.write("### Sample Properties in Database:")
            st.dataframe(potential_comps.head(5)[['address', 'structure_type', 'gla', 'year_built', 'num_beds']])
    
    # Find Comparable Properties page
    elif page == "Find Comparable Properties":
        st.write("## Find Comparable Properties")
        
        if knn_model is None or scaler is None or potential_comps is None:
            st.error("Models could not be loaded. Please check the logs.")
            return
        
        # Create property input form
        with st.form("property_form"):
            st.write("### Enter Your Property Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                structure_type = st.selectbox(
                    "Property Type", 
                    options=sorted(potential_comps['structure_type'].unique()),
                    index=0
                )
                
                gla = st.number_input(
                    "Gross Living Area (sq ft)", 
                    min_value=500, 
                    max_value=10000, 
                    value=1500
                )
                
                year_built = st.number_input(
                    "Year Built", 
                    min_value=1800, 
                    max_value=2025, 
                    value=2000
                )
            
            with col2:
                num_beds = st.selectbox(
                    "Number of Bedrooms",
                    options=[1, 2, 3, 4, 5],
                    index=2  # Default to 3 bedrooms
                )
                
                full_baths = st.selectbox(
                    "Full Bathrooms",
                    options=[0, 1, 2, 3, 4, 5],
                    index=1  # Default to 1 full bathroom
                )
                
                half_baths = st.selectbox(
                    "Half Bathrooms",
                    options=[0, 1, 2],
                    index=1  # Default to 1 half bathroom
                )
            
            submit_button = st.form_submit_button("Find Comparable Properties")
        
        # Handle form submission
        if submit_button:
            # Create bath notation
            num_baths = f"{full_baths}:{half_baths}"
            
            # Create subject property DataFrame
            subject_property = pd.DataFrame([{
                'structure_type': structure_type,
                'gla': gla,
                'year_built': year_built,
                'num_beds': num_beds,
                'num_baths': num_baths,
                'distance_km': 0.0  # Set to 0 for self
            }])
            
            # Apply scaling if needed
            numeric_features = ['gla', 'year_built', 'num_beds']
            if scaler:
                subject_property[numeric_features] = scaler.transform(subject_property[numeric_features])
            
            # Find top comps
            top_comps = get_top_comps(subject_property, potential_comps, knn_model, available_features, k=3)
            
            if top_comps.empty:
                st.warning("No comparable properties found. Try different criteria.")
            else:
                # Display results
                st.success("Found comparable properties!")
                
                # Display subject property
                st.write("### Subject Property")
                subject_display = {
                    "Property Type": structure_type,
                    "Gross Living Area": f"{gla} sq ft",
                    "Year Built": year_built,
                    "Bedrooms": num_beds,
                    "Bathrooms": f"{full_baths} full, {half_baths} half"
                }
                st.json(subject_display)
                
                # Display top comps
                st.write("### Top 3 Comparable Properties")
                
                # Create columns for the 3 properties
                cols = st.columns(3)
                
                for i, (idx, comp) in enumerate(top_comps.iterrows()):
                    with cols[i]:
                        st.subheader(f"{i+1}. {comp['address']}")
                        
                        # Property details
                        st.write(f"**Property Type:** {comp['structure_type']}")
                        st.write(f"**GLA:** {comp['gla']} sq ft")
                        st.write(f"**Year Built:** {comp['year_built']}")
                        st.write(f"**Bedrooms:** {comp['num_beds']}")
                        st.write(f"**Bathrooms:** {comp['num_baths']}")
                        
                        # Sale details if available
                        if 'sale_price' in comp and pd.notna(comp['sale_price']):
                            st.write(f"**Sale Price:** ${comp['sale_price']:,.0f}")
                        
                        if 'sale_date' in comp and pd.notna(comp['sale_date']):
                            st.write(f"**Sale Date:** {comp['sale_date']}")
                        
                        # Similarity score
                        similarity = 1.0 / (1.0 + comp['distance_score'])
                        st.progress(similarity)
                        st.write(f"**Similarity:** {similarity:.2%}")
                
                # Visual comparison
                st.write("### Visual Comparison")
                
                # Extract data for plotting
                labels = ['Subject Property'] + [f"Comp {i+1}" for i in range(len(top_comps))]
                
                # Combine subject and comps for visualization
                subject_unscaled = pd.DataFrame([{
                    'structure_type': structure_type,
                    'gla': gla,
                    'year_built': year_built,
                    'num_beds': num_beds
                }])
                
                # Get original values for comps
                comp_values = top_comps[['gla', 'year_built', 'num_beds']].copy()
                if scaler:
                    # Inverse transform if scaled
                    comp_values = pd.DataFrame(
                        scaler.inverse_transform(comp_values),
                        columns=['gla', 'year_built', 'num_beds']
                    )
                
                # Plot comparison
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                
                # GLA comparison
                gla_values = [subject_unscaled['gla'].values[0]] + comp_values['gla'].tolist()
                axs[0, 0].bar(labels, gla_values)
                axs[0, 0].set_title('Gross Living Area (sq ft)')
                axs[0, 0].tick_params(axis='x', rotation=45)
                
                # Year built comparison
                year_values = [subject_unscaled['year_built'].values[0]] + comp_values['year_built'].tolist()
                axs[0, 1].bar(labels, year_values)
                axs[0, 1].set_title('Year Built')
                axs[0, 1].tick_params(axis='x', rotation=45)
                
                # Bedroom comparison
                bed_values = [subject_unscaled['num_beds'].values[0]] + comp_values['num_beds'].tolist()
                axs[1, 0].bar(labels, bed_values)
                axs[1, 0].set_title('Number of Bedrooms')
                axs[1, 0].tick_params(axis='x', rotation=45)
                
                # Similarity comparison
                similarity_values = [1.0] + [1.0 / (1.0 + score) for score in top_comps['distance_score'].values]
                axs[1, 1].bar(labels, similarity_values)
                axs[1, 1].set_title('Similarity Score')
                axs[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # About page
    elif page == "About":
        st.write("## About the Property Comparison System")
        
        st.write("""
        This application uses machine learning to find comparable properties based on key features.
        
        ### How It Works
        
        The system uses a k-Nearest Neighbors (k-NN) algorithm to find the most similar properties based on:
        
        - **Structure Type**: The type of property (detached, townhouse, condo, etc.)
        - **Gross Living Area (GLA)**: The square footage of the property
        - **Year Built**: When the property was constructed
        - **Number of Bedrooms**: Total bedroom count
        - **Location**: Proximity to other properties
        
        ### Good Comps Are:
        
        - Sold recently (within last 90 days)
        - Same property type and structure as the subject
        - Similar features (GLA, bedrooms, bathrooms)
        - Similar neighborhood/location
        - Similar quality/condition & age
        
        ### About the Model
        
        The model has been trained on real appraisal data and achieves high accuracy in finding comparable properties.
        """)
        
        st.info("This application was developed using Streamlit, scikit-learn, and pandas.")

if __name__ == "__main__":
    main()
