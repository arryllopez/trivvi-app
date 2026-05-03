import pandas as pd
import numpy as np
import os
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(data_dir="data"):
    restaurants = pd.read_csv(f"{data_dir}/restaurants.csv")
    users = pd.read_csv(f"{data_dir}/users.csv")
    interactions = pd.read_csv(f"{data_dir}/interactions.csv")
    return restaurants, users, interactions

# 1. Collaborative Filtering - SVD

class SVDModel:

    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_index = {}
        self.item_index = {}
        self.user_mean = {}
        self.global_mean = 0.0

    def fit(self, ct_df):
        users = ct_df["user_id"].unique()
        items = ct_df["restaurant_id"].unique()
        #dict comprehension
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {u: i for i, u in enumerate(items)}

        n_users, n_items = len(users), len(items)
        self.global_mean = cf_df["rating"].mean()

        R = np.zeros((n_users, n_items))
        for _, row in cf_df.iterrows():
            R[self.user_index[row["user_id"]],
              self.item_index[row["restaurant_id"]]] = row["rating"]
        
        for uid, uidx in self.user_index.items():
            observed = R[uidx][R[uidx] != 0]
            self.user_mean[uid] = observed.mean() if len(observed) > 0 else self.global_mean
            R[uidx][R[uidx] != 0] -= self.user_mean[uid]
        
        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(csr_matrix(R), k=k)
        sigma_diag = np.diag(sigma)

        self.user_factors = U @ sigma_diag
        self.item_factors = Vt.T
        return self

    def predict(self, user_id, restaurant_id):
        bias = self.user_mean.get(user_id, self.global_mean)
        if user_id not in self.user_index or restaurant_id not in self.item_index:
            #if we've never seen user before, fall back to the global average cold start fallback
            return bias
        u = self.user_index[user_id]
        i = self.item_index[restaurant_id]
        #dot product of the user vector and restaurant vector, pointing in similar directions means the user will probably like the restaurant
        score = float(self.user_factors[u] @ self.item_factors[i]) + bias
        return np.clip(score, 1.0, 5.0) #stay between 1, 5 ratings

# MIGHT NEED READJUSTMENTS
# Viewed is just dead center 2.5
# Saved is above average, thinking a 3.2 rating
# Visited is 3.8, customer actually went to the place, thinking about 4.0?
def build_cf_model(interactions, n_factors=50, verbose=True):
    IMPLICIT_RATINGS = {"viewed": 2.5, "saved": 3.2, "visited": 3.8}

    cf_data = []
    for _, row in interactions.iterrows():
        if pd.notna(row["rating"]):
            cf_data.append((row["user_id"], row["restaurant_id"], float(row["rating"])))
        elif row["interaction_type"] in IMPLICIT_RATINGS:
            cf_data.append((row["user_id"], row["restaurant_id"],IMPLICIT_RATINGS[row["interaction_type"]]))

    cf_df = pd.DataFrame(cf_data, columns=["user_id", "restaurant_id", "rating"])
    cf_df = cf_df.groupby(["user_id", "restaurant_id"])["rating"].max().reset_index()

    if verbose:
        n_u = cf_df["user_id"].nunique()
        n_r = cf_df["restaurant_id"].nunique()
        print(f"CF training matrix: {len(cf_df)} pairs | {n_u} users × {n_r} restaurants")

    model = SVDModel(n_factors=n_factors).fit(cf_df)
    if verbose:
        print(f" SVD model trained — {n_factors} latent factors")
        
    return model, cf_df

def build_restaurant_features(restaurants):
    df = restaurants.copy()

    cuisine_dummies = pd.get_dummies(df["cuisine_type"], prefix = "cuisine")

    mlb = MultiLabelBinarizer()
    tags_split = df["tags"].str.split("|")
    tags_encoded = pd.DataFrame(
        mlb.fit_transform(tags_split),
        columns=[f"tag_{t}" for t in mlb.classes_],
        index=df.index
    )
    
    numeric = df[["price_range", "avg_rating", "ttc_accessible", "halal_certified", "kosher_certified"]].astype(float)
    
    scaler = MinMaxScaler() # make the price_range and avg_rating to 0-1 so they don't overpower anything else
    
    numeric[["price_range", "avg_rating"]] = scaler.fit_transform(numeric[["price_range", "avg_rating"]])

    feature_matrix = pd.concat(
        [df[["restaurant_id"]].reset_index(drop=True),
        cuisine_dummies.reset_index(drop=True),
        tags_encoded.reset_index(drop=True),
        numeric.reset_index(drop=True)],
        axis=1
    ).set_index("restaurant_id")
    
    return feature_matrix, mlb, scaler
    
def build_user_profile(user_id, users, interactions, feature_matrix):
    user_row = users[users["user_id"] == user_id]
    if user_row.empty:
        return None
    user_row = user_row.iloc[0]

    WEIGHTS = {"rated": 1.0, "visited": 0.8, "saved": 0.5, "viewed": 0.2}

    user_ints = interactions[interactions["user_id"] == user_id].copy()
    user_ints["weight"] = user_ints.apply(
        lambda r: (r["rating"] / 5.0) if pd.notna(r["rating"]) else WEIGHTS.get(r["interaction_type"], 0.2),axis = 1)
    
    interacted_ids = user_ints["restaurant_id"].values
    valid_ids = [rid for rid in interacted_ids if rid in feature_matrix.index]

    if valid_ids:
        weights = user_ints[user_ints["restaurant_id"].isin(valid_ids)]["weight"].values
        vectors = feature_matrix.loc[valid_ids].values
        profile = np.average(vectors, axis=0, weights=weights)
    else:
        profile = np.zeros(feature_matrix.shape[1])
    
    preferred_cuisines = user_row["preferred_cuisines"].split("|")
    for cuisine in preferred_cuisines:
        col = f"cuisine_{cuisine}"
        if col in feature_matrix.columns:
            idx = feature_matrix.columns.get_loc(col)
            profile[idx] = min(profile[idx] + 0.3, 1.0)
    diet = user_row["dietary_restrictions"]
    transit = user_row["transit_dependent"]

    # vegan and vegetarian are hard requirements
    if diet == "vegan":
        if "tag_vegan-friendly" in feature_matrix.columns:
            profile[feature_matrix.columns.get_loc("tag_vegan-friendly")] = min(
                profile[feature_matrix.columns.get_loc("tag_vegan-friendly")] + 0.4, 1.0)
    elif diet == "vegetarian":
        if "tag_vegetarian-friendly" in feature_matrix.columns:
            profile[feature_matrix.columns.get_loc("tag_vegetarian-friendly")] = min(
                profile[feature_matrix.columns.get_loc("tag_vegetarian-friendly")] + 0.3, 1.0)
    elif diet == "halal":
        if "halal_certified" in feature_matrix.columns:
            profile[feature_matrix.columns.get_loc("halal_certified")] = 1.0
    elif diet == "kosher":
        if "kosher_certified" in feature_matrix.columns:
            profile[feature_matrix.columns.get_loc("kosher_certified")] = 1.0

    if transit:
        if "ttc_accessible" in feature_matrix.columns:
            profile[feature_matrix.columns.get_loc("ttc_accessible")] = 1.0

    if "price_range" in feature_matrix.columns:
        profile[feature_matrix.columns.get_loc("price_range")] = (user_row["price_sensitivity"] - 1) / 3.0

    return profile

def content_based_scores(user_profile, feature_matrix):
    rest_vectors = feature_matrix.values.astype(np.float64)
    norm_profile = np.linalg.norm(user_profile)
    norm_rests = np.linalg.norm(rest_vectors, axis=1) # calculate length of every restaurant vector row by row

    denom = norm_profile * norm_rests
    denom[denom == 0] = 1e-9 # replace zeros with extremely tiny number since if denom 0, error occurs.

    similarities = (rest_vectors @ user_profile) / denom # divide by lengths to normalize and give a value between 0 and 1
    return pd.Series(similarities, index=feature_matrix.index)


# 2. Content-Based Filtering
