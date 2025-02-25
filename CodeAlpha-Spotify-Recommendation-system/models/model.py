import torch.nn as nn
import torch.nn.functional as F
import torch

class MusicRecommendationModel(nn.Module):
    """
    A neural network model for music recommendation.
    Args:
        num_users (int): The number of unique users.
        num_songs (int): The number of unique songs.
        embedding_dim (int, optional): The dimension of the embedding vectors. Default is 10.
    Attributes:
        user_embedding (nn.Embedding): Embedding layer for users.
        song_embedding (nn.Embedding): Embedding layer for songs.
        fc1 (nn.Linear): Fully connected layer that takes concatenated embeddings and outputs 64 features.
        fc2 (nn.Linear): Fully connected layer that outputs a single score.
    Methods:
        forward(users, songs):
            Performs a forward pass of the model.
            Args:
                users (torch.Tensor): Tensor containing user indices.
                songs (torch.Tensor): Tensor containing song indices.
            Returns:
                torch.Tensor: The predicted score for each user-song pair.
    """
    def __init__(self, num_users, num_songs, embedding_dim=10):
        super(MusicRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, users, songs):
        user_embeds = self.user_embedding(users)
        song_embeds = self.song_embedding(songs)
        x = torch.cat([user_embeds, song_embeds], dim=1)  # Ensure dim=1 for concatenation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x