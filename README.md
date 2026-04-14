# movie-recommendation-engine
# 🎬 AI-Powered Movie Recommendation System

### Content-Based • Collaborative • Hybrid Filtering with Interactive UI

---

## Overview

This project is a complete **Movie Recommendation System** built using multiple AI techniques:

- Content-Based Filtering (TF-IDF + Cosine Similarity)
- Collaborative Filtering (Matrix Factorization using SVD)
- Hybrid Recommendation System (Combined Intelligence)
- Interactive Web App using Streamlit

The system recommends movies based on:
- Similar content (genres)
- User preferences
- Combined hybrid scoring

---

##  Key Features

- earch movies and get similar recommendations
- Personalized recommendations based on user ID
- Hybrid system combining both approaches
- Rating-based visualization charts
- Fast and scalable recommendation engine
-  Beautiful Streamlit UI

---

##  Recommendation Techniques

###  1. Content-Based Filtering
- Uses **TF-IDF vectorization** on movie genres  
- Computes similarity using **cosine similarity**  
- Recommends movies with similar content  

---

###  2. Collaborative Filtering
- Uses **Matrix Factorization (SVD)**  
- Learns hidden patterns in user ratings  
- Predicts user preferences  

---

###  3. Hybrid System
- Combines both methods using weighted scoring  

\[
Score = \alpha \cdot Content + (1 - \alpha) \cdot Collaborative
\]

---

## Dataset

- Source: MovieLens (ml-latest-small)
- Movies: ~9,742
- Ratings: ~100,836
- Users: ~610

---

##Project Structure
MOVIE_RECOMMENDATION/
│
├── movie_recommender_part1.py # Content-Based Filtering
├── movie_recommender_part2.py # Collaborative + Hybrid + UI
├── movies.csv
├── ratings.csv


---

##  Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- TF-IDF Vectorizer  
- Truncated SVD  

---

##  Streamlit Web App

Run the application:

```bash
streamlit run movie_recommender_part2.py
