from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from recommender import TrivviRecommender, load_data

app = FastAPI(
    title="Trivvi API",
    description="Restaurant recommendation engine for Toronto",
    version="0.1.0"
)

print("Loading Trivvi model...")
rec = TrivviRecommender.load("model")
restaurants, users, interactions = load_data()
print("Ready")

class RestaurantOut(BaseModel):
    restaurant_id: str
    name: str
    neighbourhood: str
    cuisine_type: str
    price_range: int
    avg_rating: float
    tags: list[str]
    ttc_accessible: bool
    halal_certified: bool
    kosher_certified: bool

class RecommendationOut(BaseModel):
    restaurant_id: str
    name: str
    neighbourhood: str
    cuisine_type: str
    price_range: int
    avg_rating: float
    tags: list[str]
    ttc_accessible: bool
    halal_certified: bool
    kosher_certified: bool
    hybrid_score: float
    cf_score: float
    cb_score: float

class UserOut(BaseModel):
    user_id: str
    age_group: str
    preferred_cuisines: list[str]
    price_sensitivity: int
    neighbourhood: str
    dietary_restrictions: str
    transit_dependent: bool