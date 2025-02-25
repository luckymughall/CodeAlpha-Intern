# 🎵 Music Recommendation System

A PyTorch-based Spotify music recommendation system that predicts and suggests songs to users based on their listening history. It utilizes collaborative filtering with neural embeddings.

## 🚀 Features

- Trainable **PyTorch model** for music recommendations  
- **User & song embeddings** for improved accuracy  
- Supports **real-world datasets** (e.g., `music_datasets.csv`)  
- **Retrieves song metadata** (Title, Artist, Album)  
- Scalable for large datasets  

## 📂 Project Structure

📦 Music-Recommendation-System
│── data/datasets.py  # Loads and processes dataset
│── modls/model.py # Defines the recommendation model
│── utils/utils.py # Utility functions for making recommendations
│── train.ipynb # Jupyter Notebook for training & evaluation
│── data/music_datasets.csv # Sample dataset with user-song interactions
│── README.md # Project documentation
│── Requirements.txt # Requirements for packages for the project

## 📥 Dataset

This project uses a dataset containing **user interactions with songs**.

**Dataset Columns:**

| Column    | Description |
|-----------|------------|
| `Username` | User who listened to the song |
| `Artist`   | Song's artist |
| `Track`    | Song's title |
| `Album`    | Album name |
| `Date`     | Date of interaction |
| `Time`     | Time of interaction |

**Processing Steps:**

1. Convert `Username` into `user_id`  
2. Convert `(Track, Artist)` into `song_id`  
3. Assign a **play count** (each row = 1 play)  

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/Elabs-llc/CodeAlpha_Music_Recomendation_System.git
cd Music_Recommendation_System
```

### 2️⃣  Install necessary modules

```sh
pip install torch pandas scikit-learn jupyter
```

### 3️⃣ Run Jupyter Notebook

```sh
jupyter notebook
```

Open train.ipynb to train the model and make predictions.

## 📝 License

This project is open-source under the MIT License.
