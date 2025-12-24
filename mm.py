# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ### **1. DATA LOADING AND SETUP**

# %%
df = pd.read_csv('data/players_20.csv')

# %%
df.info()

# %%
for column in df.columns:
    num_missing = df[column].isnull().sum()
    if num_missing > 0:
        print(f"Column '{column}' has {num_missing} missing values.")

# %%


# %% [markdown]
# *player_tags*, *position* and *jersey number* give a kind of obvious idea of the position played by the player. I intend to cluster classify these players based on their playing attributes like speed, passing, shooting, dribbling abilities etc.
# 
# things like club being played for, club loaned or bought from are out of the question. So I will limit my exploration to the following columns:
# 
#     - those that include phisicali/body
#     - attacking/defensive/technical abilities
#     - mental/tactical/movement

# %%
features = [
    # keeping player positions for filtering later (won't be used to train)
    # "player_positions",

    # Physical / Body
    "age",
    "height_cm",
    "weight_kg",

    # Aggregate Outfield Attributes
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",

    # Attacking
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",

    # Technical Skills
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",

    # Movement
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",

    # Power / Physicality
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",

    # Mental / Tactical
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",

    # Defensive
    "defending_marking",
    "defending_standing_tackle",
    "defending_sliding_tackle",

    # Goalkeeping
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes"
]

# %%
data = df[features]
for column in data.columns:
    num_missing = data[column].isnull().sum()
    if num_missing > 0:
        print(f"Column '{column}' has {num_missing} missing values.")
data.head()

# %%
# to see how many records I have (for dealing with the missing data):
data.shape

# %%
# data.dropna(inplace=True)
# using imputation for missing outfield attributes for goalkeepers
# data.fillna(mean, inplace=True)
# filling na with mean values
data.fillna(data.mean(), inplace=True)

# %%
data.shape

# %%
for column in data.columns:
    num_missing = data[column].isnull().sum()
    if num_missing > 0:
        print(f"Column '{column}' has {num_missing} missing values.")

# %% [markdown]
# ### **2. EXPLORATORY DATA ANALYSIS**

# %%
data.describe().T

# %% [markdown]
# Visualizing distributions for outfield players' attributes

# %%
key_features = ["pace", "shooting", "passing", "defending", "physic"]

plt.figure(figsize=(15, 8))

for i, col in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    plt.hist(data[col].values, bins=30)
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# %% [markdown]
# Visualizing distributions for key goalkeeping attributes

# %%
gk_features = [
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes"
]

plt.figure(figsize=(15, 8))

for i, col in enumerate(gk_features, 1):
    plt.subplot(2, 3, i)
    plt.hist(data[col].values, bins=30)
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# %% [markdown]
# It makes sense that the distributions show some trend from 0-20, and then another somehow normally distributed around 50-90. 
# 
# Those from 0-20 show the ratings for outfield players in goalkeeping positions wchich is quite expected.
# 
# That being the case, it is expected that the precision and recall for the cluster with GK to be higher.

# %%
# data = data[data['player_positions'] != 'GK']
# data.drop(columns=['player_positions'], inplace=True)
data.head()

# %% [markdown]
# ### **3. DATA PREPOCESSING**

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()


# %% [markdown]
# The explained variance curve proves the attributes are not random, it shows the data has a particular structure 
# 
# It has also justified dimensionality reduction as ~85% variance explained by ~10 components”

# %%
X_pca.shape

# %%
from sklearn.cluster import KMeans

inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Choosing k")
plt.grid(True)
plt.show()

# %% [markdown]
# Choosing K=4 based on domain knowledge and support from the elbow graph

# %%
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores for Different k")
plt.grid(True)
plt.show()

# %% [markdown]
# Silhouette analysis indicated the highest score at k = 2, suggesting a strong separation between goalkeepers and outfield players. 
# 
# However, given the objective of identifying four primary playing roles, k = 4 was selected. 
# 
# The reduced silhouette score at k = 4 reflects the known overlap between outfield positions, which is consistent with football **domain knowledge**.

# %% [markdown]
# ### **4. MODEL TRAINING**

# %%
from sklearn.cluster import KMeans

k = 4

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

# %%
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
cluster_counts

# %%
data_clustered = data.copy()
data_clustered["cluster"] = cluster_labels

# %%
data_clustered.head()

# %%
profile_features = [
    "pace", "shooting", "passing", "defending", "physic",
    "goalkeeping_diving", "goalkeeping_reflexes"
]

data_clustered.groupby("cluster")[profile_features].mean()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    s=8,
    alpha=0.6
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clusters (k=4) in PCA Space")
plt.colorbar(label="Cluster")
plt.show()

# %%
# 1. Create Human-Readable Axes
# We take the average of key attributes to create a "Composite Score"
data['Offensive_Skill'] = (data['shooting'] + data['passing'] + data['dribbling']) / 3
data['Defensive_Physicality'] = (data['defending'] + data['physic']) / 2

# 2. Plot with these new clear axes
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    data['Offensive_Skill'], 
    data['Defensive_Physicality'], 
    c=cluster_labels, 
    cmap='viridis', 
    s=15, 
    alpha=0.6
)

# 3. Label with Actual Football Terms (No abstract "PCs")
plt.xlabel("Offensive Skill Rating (0-100)")
plt.ylabel("Defensive & Physical Rating (0-100)")
plt.title("Player Roles: Offense vs. Defense")
plt.colorbar(scatter, label="Cluster ID")
plt.grid(True, linestyle='--', alpha=0.5)

# 4. Add rough text labels to help the marker (Adjust positions based on your plot)
plt.text(80, 20, "Attackers", fontsize=12, fontweight='bold', color='black')
plt.text(20, 80, "Defenders", fontsize=12, fontweight='bold', color='black')
plt.text(70, 70, "Midfielders", fontsize=12, fontweight='bold', color='black')

plt.show()

# %%



