import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure the .csv is in the same local directory
dt = pd.read_csv("test.csv")
# Load into Dataframe
df = pd.DataFrame(data=dt, columns = ["test_id","description_x","description_y","same_security"])

# Extract the individual descriptions

# Turn into TFIDF Vectors
vec = TfidfVectorizer()
xVec = vec.fit_transform(df['description_x'])
yVec = vec.transform(df['description_y'])

# Update the same_security column with the cosine similarity data
for n in range (df.shape[0] + 1):
    df.set_value(n, 'same_security', cosine_similarity(xVec[n], yVec[n]))

# Export to file
df.to_csv("results.csv", encoding='utf-8')