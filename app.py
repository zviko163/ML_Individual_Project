import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the "Brain" and Database
@st.cache_resource
def load_resources():
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    data = pd.read_csv('clustered_fifa_data.csv')
    return kmeans, scaler, data

kmeans, scaler, df = load_resources()

# FEATURE LIST (Must match training EXACTLY)
# Note: Ensure you use the same columns here as you did in mm.py for fitting!
features = [
    "pace", "shooting", "passing", "dribbling", "defending", "physic",
    "goalkeeping_diving", "goalkeeping_reflexes"
]

st.title("⚽ FIFA Intelligent Scouting System")

# TABS: Separate the "Lookup" logic from the "Prediction" logic
tab1, tab2 = st.tabs(["🔄 Find Replacements", "🆕 Scout Custom Player"])

# ==========================================
# TAB 1: OFFLINE INFERENCE (Fast Lookup)
# ==========================================
with tab1:
    st.header("Transfer Market Recommender")
    st.write("Replace an existing player with a cheaper alternative from the same cluster.")
    
    selected_player = st.selectbox("Select a Player:", df['short_name'].sort_values().unique())
    n_recs = st.slider("Recommendations:", 1, 10, 5)
        
    if st.button("Find Alternatives"):
            # 1. ROBUST LOOKUP
            # Find all players with this name
            target_rows = df[df['short_name'] == selected_player]
            
            if target_rows.empty:
                st.error(f"Player '{selected_player}' not found in FIFA 20 database.")
            else:
                # HANDLE DUPLICATES: Pick the one with the highest 'overall' rating
                # This ensures if you pick "Danilo", you get the Juventus one, not a random reserve.
                target = target_rows.sort_values('overall', ascending=False).head(1)
                
                # Now we safely extract the single cluster and single vector
                target_cluster = target['cluster'].values[0]
                target_stats = target[features].values # Shape is now guaranteed (1, 8)
                
                # 2. Filter & Calculate Distance
                candidates = df[df['cluster'] == target_cluster].copy()
                candidate_stats = candidates[features].values
                
                distances = np.linalg.norm(candidate_stats - target_stats, axis=1)
                candidates['Similarity_Distance'] = distances
                
                # 3. Show Results
                results = candidates.sort_values('Similarity_Distance').iloc[1:n_recs+1]
                
                st.success(f"Players most similar to {selected_player} (Rated {target['overall'].values[0]}):")
                st.dataframe(results[['short_name', 'overall', 'value_eur', 'Similarity_Distance']])

# ==========================================
# TAB 2: ONLINE INFERENCE (Context-Aware Prediction)
# ==========================================
with tab2:
    st.header("Scout a New Prospect")
    st.write("Enter raw stats to classify a player who isn't in the database.")
    
    col1, col2 = st.columns(2)
    
    # 1. USER INPUTS (The 8 Key Stats)
    input_stats = {}
    with col1:
        input_stats['pace'] = st.slider("Pace", 0, 100, 70)
        input_stats['shooting'] = st.slider("Shooting", 0, 100, 60)
        input_stats['passing'] = st.slider("Passing", 0, 100, 65)
        input_stats['dribbling'] = st.slider("Dribbling", 0, 100, 70)
    with col2:
        input_stats['defending'] = st.slider("Defending", 0, 100, 50)
        input_stats['physic'] = st.slider("Physical", 0, 100, 60)
        input_stats['goalkeeping_diving'] = st.slider("GK Diving", 0, 100, 10)
        input_stats['goalkeeping_reflexes'] = st.slider("GK Reflexes", 0, 100, 10)
        
    if st.button("Predict Role"):
        # ---------------------------------------------------------
        # SMART IMPUTATION LOGIC
        # ---------------------------------------------------------
        
        # A. Define the full 43-feature list (Must match training order)
        full_features = [
            "age", "height_cm", "weight_kg",
            "pace", "shooting", "passing", "dribbling", "defending", "physic",
            "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy", "attacking_short_passing", "attacking_volleys",
            "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing", "skill_ball_control",
            "movement_acceleration", "movement_sprint_speed", "movement_agility", "movement_reactions", "movement_balance",
            "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots",
            "mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
            "defending_marking", "defending_standing_tackle", "defending_sliding_tackle",
            "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning", "goalkeeping_reflexes"
        ]
        
        # B. Create Two Templates (GK vs Outfield)
        # We split the database to find the "Average GK" and "Average Outfielder"
        # (We assume Cluster 3 is GKs based on your earlier analysis, 
        #  BUT to be safe, we just filter by position name or stats)
        
        # Heuristic: Players with GK Diving > 50 are Goalkeepers
        gk_df = df[df['goalkeeping_diving'] > 50]
        outfield_df = df[df['goalkeeping_diving'] <= 50]
        
        avg_gk = gk_df[full_features].mean()
        avg_outfield = outfield_df[full_features].mean()
        
        # C. Decide which template to use based on User Input
        # If the user put high GK stats, they are describing a GK.
        if input_stats['goalkeeping_diving'] > 40:
            base_vector = avg_gk.copy()
            st.caption("ℹ️ Detected Goalkeeper stats: Filling missing attributes with GK averages.")
        else:
            base_vector = avg_outfield.copy()
            st.caption("ℹ️ Detected Outfield stats: Filling missing attributes with Outfield averages.")
            
        # D. Overwrite with User Inputs
        base_vector['pace'] = input_stats['pace']
        base_vector['shooting'] = input_stats['shooting']
        base_vector['passing'] = input_stats['passing']
        base_vector['dribbling'] = input_stats['dribbling']
        base_vector['defending'] = input_stats['defending']
        base_vector['physic'] = input_stats['physic']
        base_vector['goalkeeping_diving'] = input_stats['goalkeeping_diving']
        base_vector['goalkeeping_reflexes'] = input_stats['goalkeeping_reflexes']
        
        # ---------------------------------------------------------
        # PREDICT
        # ---------------------------------------------------------
        # Reshape and Scale
        raw_values_reshaped = base_vector.values.reshape(1, -1)
        scaled_values = scaler.transform(raw_values_reshaped)
        
        # Predict Cluster
        predicted_cluster = kmeans.predict(scaled_values)[0]
        st.info(f"The AI Model predicts this player belongs to **Cluster {predicted_cluster}**")
        
        # Find Nearest Neighbor
        # We scale the whole DB to match the input scale
        db_vectors = scaler.transform(df[full_features].values)
        dist = np.linalg.norm(db_vectors - scaled_values, axis=1)
        
        closest_idx = np.argmin(dist)
        closest_player = df.iloc[closest_idx]
        
        st.markdown(f"### Closest Match: **{closest_player['short_name']}**")
        st.write(f"Role: {closest_player.get('player_positions', 'Unknown')}")