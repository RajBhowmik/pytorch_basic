import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load the CSV file
file_path = '/Users/rajbhowmik/python_code/pytorch_basic/data.csv'
df = pd.read_csv(file_path)

# Initialize the sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Lowercase all attributes for consistency
attributes = [attr.lower() for attr in df['Attribute'].unique()]
embeddings = model.encode(attributes, convert_to_tensor=True)

# Function to find synonyms within the column
def find_synonyms(attribute, embeddings, attributes, top_k=5, threshold=0.5):
    query_embedding = model.encode(attribute, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    
    # Find top_k results and filter by similarity threshold
    top_results = torch.topk(cos_scores, k=len(attributes))
    synonyms = []
    for score, idx in zip(top_results[0], top_results[1]):
        if attributes[idx] != attribute and score.item() >= threshold:  # Exclude the attribute itself and apply threshold
            synonyms.append(attributes[idx])
        if len(synonyms) >= top_k:
            break
    return synonyms

# Create a dictionary to store the synonyms
synonym_dict = {}

# Find synonyms for each attribute
for attribute in attributes:
    synonym_dict[attribute] = find_synonyms(attribute, embeddings, attributes, top_k=5, threshold=0.5)

# Print the synonym dictionary
print(synonym_dict)

# Optionally, save the dictionary to a CSV file
synonym_df = pd.DataFrame(list(synonym_dict.items()), columns=['Attribute', 'Synonyms'])
synonym_df['Synonyms'] = synonym_df['Synonyms'].apply(lambda x: ', '.join(x))
output_file_path = 'synonym_dictionary.csv'
synonym_df.to_csv(output_file_path, index=False)

print(synonym_df)