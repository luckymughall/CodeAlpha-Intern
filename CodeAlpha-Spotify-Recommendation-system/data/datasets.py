import pandas as pd
import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    """
    A custom PyTorch Dataset for music recommendation data.

    Attributes:
        users (torch.Tensor): Tensor containing user IDs.
        songs (torch.Tensor): Tensor containing song IDs.
        play_counts (torch.Tensor): Tensor containing play counts.

    Methods:
        __init__(df):
            Initializes the MusicDataset with user IDs, song IDs, and play counts.
            Args:
                df (pandas.DataFrame): DataFrame containing the columns 'user_id', 'song_id', and 'play_count'.

        __len__():
            Returns the number of samples in the dataset.
            Returns:
                int: Number of samples in the dataset.

        __getitem__(idx):
            Returns the user ID, song ID, and play count for the given index.
            Args:
                idx (int): Index of the sample to retrieve.
            Returns:
                tuple: (user_id, song_id, play_count) for the given index.

        load_data(filepath):
            Loads and preprocesses data from a CSV file.
            Args:
                filepath (str): Path to the CSV file containing the dataset.
            Returns:
                pandas.DataFrame: Preprocessed DataFrame with 'user_id', 'song_id', and 'play_count' columns.
    """
    
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.songs = torch.tensor(df['song_id'].values, dtype=torch.long)
        self.play_counts = torch.tensor(df['play_count'].values, dtype=torch.float32) 

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.songs[idx], self.play_counts[idx]
    
    # Load and preprocess data
    def load_data(filepath="data/music_datasets.csv"):
        df = pd.read_csv(filepath)

        # Convert categorical data into numerical IDs
        df['user_id'] = df['Username'].astype('category').cat.codes
        df['song_id'] = df[['Track', 'Artist']].astype(str).apply(lambda x: ' - '.join(x), axis=1)
        df['song_id'] = df['song_id'].astype('category').cat.codes

        # Since each row represents one play, set play_count to 1
        df['play_count'] = 1

        return df








