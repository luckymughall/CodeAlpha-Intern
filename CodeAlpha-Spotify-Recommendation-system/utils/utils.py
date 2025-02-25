import torch

# def recommend_songs_for_user(model, user_id, song_ids, top_k=10):
#     model.eval()
#     with torch.no_grad():

#         # create tensors for users and songs
#         user = torch.tensor([user_id] * len(song_ids), dtype=torch.long)
#         #song = torch.tensor([song_ids], dtype=torch.long)
#         song = torch.tensor(song_ids, dtype=torch.long)  # Remove the extra list wrapping


#         ## Ensure both user and song tensor have an additional batch dimension
#         # user = user.unsqueeze(0)
#         # song = song.unsqueeze(0)

#         # Debugging the tensors
#         print("Initial user tensor shape:  ", user.shape)
#         print("Initial song tensor shape:  ", song.shape)

#         # concatebate or process inputs as expected by the model
#         #input_tensor  = torch.cat((user, song), dim=1)

#         # debuggung the input tensor
#         #print("Input tensor shape:  ", input_tensor.shape)


#         # pass the inputes through the model
#         predictions = model(user, song)

#         # debuggung the predictions
#         print("Predictions tensor shape:  ", predictions.shape) 

#         predictions = predictions.squeeze()

#         # get the top n song indeicies with the highest predictions
#         indices = predictions.argsort(descending=True)[:top_k]
#         recommended_sonds_ids = [song_ids[i] for i in indices]
#         return recommended_sonds_ids
    

"""
Recommend top-k songs for a given user based on the model's predictions.
Args:
    model (torch.nn.Module): The trained recommendation model.
    user_id (int): The ID of the user for whom to recommend songs.
    df (pandas.DataFrame): DataFrame containing song information with columns 'song_id', 'Track', 'Artist', and 'Album'.
    top_k (int, optional): The number of top recommendations to return. Defaults to 10.
Returns:
    pandas.DataFrame: DataFrame containing the recommended songs with columns 'Track', 'Artist', and 'Album'.
"""

def recommend_songs_for_user(model, user_id, df, top_k=10):
    model.eval()
    with torch.no_grad():
        song_ids = df['song_id'].unique()

        # Create tensors for users and songs
        user = torch.tensor([user_id] * len(song_ids), dtype=torch.long)
        song = torch.tensor(song_ids, dtype=torch.long)

        # Predict scores
        predictions = model(user, song).squeeze()
        indices = predictions.argsort(descending=True)[:top_k]

        # Retrieve song details
        recommended_songs = df[['Id', 'Track', 'Artist', 'Album']].iloc[indices.tolist()].drop_duplicates()

        return recommended_songs



# get user name from user id
def get_user_name(df, user_id):
    return df[df['user_id'] == user_id]['Username'].values[0]

















