# %% [markdown]
# # Books Recommendation System

# %%
# ignore warnings especially for missing cuda driver tensorflow
import warnings
warnings.filterwarnings('ignore')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# %% [markdown]
# ## Data Understanding

# %%
# load books dataset
df_books = pd.read_csv("dataset/Books.csv", low_memory=False, encoding="utf-8")
df_books.head(3)

# %%
# load ratings dataset
df_ratings = pd.read_csv("dataset/Ratings.csv")
df_ratings.head(3)

# %%
# load users dataset
df_users = pd.read_csv("dataset/Users.csv")
df_users.head(3)

# %%
df_books.shape, df_ratings.shape, df_users.shape

# %% [markdown]
# Now we want to find some information from the dataset, like data type, checking null values, doing some analysis, and EDA.

# %% [markdown]
# ### Books

# %%
df_books.info()

# %%
df_books["ISBN"].nunique(), df_books["Book-Author"].nunique(), df_books["Publisher"].nunique()

# %% [markdown]
# We know that there are 271.360 unique books, 1022.022 unique authors, and 16.807 unique publisher.

# %%
# display top publication years
df_books["Year-Of-Publication"].value_counts().head(10)

# %%
# display top book authors
df_books["Book-Author"].value_counts().head(10)

# %%
# display top publishers
df_books["Publisher"].value_counts().head(10)

# %%
df_int_publication = df_books[df_books["Year-Of-Publication"].astype(str).str.isnumeric()]
df_int_publication["Year-Of-Publication"] = pd.to_datetime(df_int_publication["Year-Of-Publication"], format="%Y", errors='coerce')

current_year = datetime.now().year
books = df_int_publication[(df_int_publication["Year-Of-Publication"] >= pd.to_datetime('1990')) & (df_int_publication["Year-Of-Publication"] <= pd.to_datetime(current_year, format='%Y'))]

books_per_year = books.groupby(books["Year-Of-Publication"].dt.year).size()

plt.figure(figsize=(10, 6))
books_per_year.plot(kind='bar', color='skyblue')
plt.title('Total Books Published (1990 - {})'.format(current_year))
plt.xlabel('Year')
plt.ylabel('Total Books Published')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("docs/book-published.png")
plt.show()

# %% [markdown]
# ### Ratings

# %%
df_ratings.info()

# %%
df_ratings.describe()

# %% [markdown]
# From the table above, we know that the average rating is 2.8, the minimum rating is 0 and the maximum rating is 10.

# %%
df_ratings["Book-Rating"].value_counts()

# %%
df_ratings["Book-Rating"].value_counts().plot(kind='bar', color='skyblue', title='Book Rating Distribution')
plt.xticks(rotation=0)
plt.savefig("docs/book-rating.png")

# %% [markdown]
# From the illustration above, most of the rating is 0, and then followed by 8 and 10.

# %%
df_ratings.groupby("User-ID")["Book-Rating"].count().sort_values(ascending=False).head(10)

# %% [markdown]
# From the data above, we can see that the top user that gives rating is user with ID 11676.

# %%
top_10_most_ratings = df_ratings.groupby("ISBN")["Book-Rating"].count().sort_values(ascending=False).reset_index().head(10)
top_10_most_ratings = top_10_most_ratings.join(df_books.set_index("ISBN"), on="ISBN")
top_10_most_ratings[["Book-Rating", "ISBN", "Book-Title", "Book-Author"]]


# %% [markdown]
# From the table above, we know that the top 10 book that received most rating. The "Wild Animus" book received 2502 rating from users.

# %% [markdown]
# ### Users

# %%
df_users.info()

# %%
df_users.dropna().describe()

# %% [markdown]
# From the table above, we konw that the user age average is about 34 years. However the max year and the minimum years seem to have an outlier that should be removed.

# %%
df_users["User-ID"].nunique()

# %% [markdown]
# We know that the there are 278.858 difference users from the dataset.

# %%
locations = df_users['Location'].str.split(',\s*', expand=True).stack()

location_counts = locations.value_counts()

top_locations = location_counts.head(10)

top_locations.plot(kind='bar', figsize=(10, 6))
plt.title('Top 10 User Locations')
plt.xlabel('Location')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig("docs/user-location.png")
plt.show()

# %% [markdown]
# From the image above, we know that most of the users are came from USA, Canada, and California. After that we get users from UK and Germany.

# %% [markdown]
# ## Data Preparation

# %%
# drop null values
df_books = df_books.dropna()
df_ratings = df_ratings.dropna()
df_users = df_users.dropna()

# %%
# drop duplicates values
df_books = df_books.drop_duplicates(subset=["ISBN"])
df_users = df_users.drop_duplicates(subset=["User-ID"])

# %%
# drop outlier age values in user using IQR
Q1 = df_users["Age"].quantile(0.25)
Q3 = df_users["Age"].quantile(0.75)
IQR = Q3 - Q1
df_users = df_users[~((df_users["Age"] < (Q1 - 1.5 * IQR)) | (df_users["Age"] > (Q3 + 1.5 * IQR)))]

# %% [markdown]
# Next, we will need to merge all of the data, from books, ratings, and users together.

# %%
all_df = pd.merge(df_ratings, df_books, on="ISBN", how="inner")
all_df = pd.merge(all_df, df_users, on="User-ID", how="inner")
all_df.head(3)

# %%
all_df = all_df[["User-ID", "ISBN", "Book-Rating", "Book-Title"]]
all_df.head(3)

# %%
all_df.shape    

# %%
all_df.isna().sum()

# %%
all_df.duplicated().sum()

# %% [markdown]
# We already have the final dataset that contain 748.401 row data.

# %% [markdown]
# ## Modeling

# %% [markdown]
# ### Content-Based Filtering

# %% [markdown]
# We will try to use TFIDF and Cosine Similarty to give book recommendation based on Book Title. But we will limit the data because it's very heavy and consume a lot of memory.

# %%
df_cb_books = df_books[["ISBN", "Book-Title", "Book-Author"]]
df_cb_books = df_cb_books.head(20000)

# %%
# drop duplicates values
df_cb_books.drop_duplicates(subset="Book-Title", keep="first", inplace=True)

# %%
# drop null values
df_cb_books = df_cb_books.dropna()

# %%
# create tfidf vectorizer
tfidf = TfidfVectorizer()

# %%
# fit and transform the vectorizer
tfidf_matrix = tfidf.fit_transform(df_cb_books['Book-Author'])
tfidf_matrix.shape

# %%
# create cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix) 

# %%
# create df for cosine similarity matrix
cosine_sim_df = pd.DataFrame(cosine_sim, index=df_cb_books['Book-Author'], columns=df_cb_books['Book-Title'])

# %% [markdown]
# We will create function to retrive book that are similar by using the cosine similarity matrix.

# %%
def books_recommendations(
    book_name,
    similarity_data=cosine_sim_df,
    items=df_cb_books[["Book-Title", "Book-Author"]],
    k=5,
):
    index = similarity_data.loc[:, book_name].to_numpy().argpartition(range(-1, -k, -1))

    closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]

    closest = closest.drop(book_name, errors="ignore")

    return pd.DataFrame(closest).merge(items).head(k)

# %% [markdown]
# Now we will find some book and try to get the recommendation.

# %%
df_cb_books[df_cb_books["Book-Title"] == "Hearts in Atlantis"]

# %%
books_recommendations("Hearts in Atlantis")

# %% [markdown]
# Nice, we successfully get some of book recommendation from the same author.

# %% [markdown]
# ### Collaborative Filtering

# %% [markdown]
# Now we will use collaborative filtering that use user rating to recommend the books.

# %%
all_df.head()

# %% [markdown]
# Next we will encode the user id and book ISBN.

# %%
user_ids = all_df["User-ID"].unique().tolist()
encode_user = {id: i for i, id in enumerate(user_ids)}

book_ids = all_df["ISBN"].unique().tolist()
encode_book = {id: i for i, id in enumerate(book_ids)}

# %%
data = all_df.copy()

# %%
data["user"] = data["User-ID"].map(encode_user)
data["book"] = data["ISBN"].map(encode_book)
data["rating"] = data["Book-Rating"].apply(lambda x: x / 10)

# %%
data = data[["user", "book", "rating"]]
data.head()

# %%
num_users = data["user"].nunique()
num_books = data["book"].nunique()
num_users, num_books

# %% [markdown]
# After that, we will split it into training and testing.

# %%
X = data[["user", "book"]].values
y = data["rating"].values

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# After that, we will build a recommender class for this recommendation problem.

# %%
class BookRecommender(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size, **kwargs):
        super(BookRecommender, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        self.user_embedding = keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = keras.layers.Embedding(num_users, 1)
        self.book_embedding = keras.layers.Embedding(
            num_books,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.book_bias = keras.layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        
        dot_user_book = tf.tensordot(user_vector, book_vector, 2)
        x = dot_user_book + user_bias + book_bias
        return tf.nn.sigmoid(x)

# %% [markdown]
# We will compile the model using loss binary crossentropy, with Adam optimizer, and RMSE metrics.

# %%
model = BookRecommender(num_users, num_books, 128)
 
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# %% [markdown]
# Fit the model

# %%
history = model.fit(
    x = X_train,
    y = y_train,
    batch_size = 1024,
    epochs = 10,
    validation_data = (X_test, y_test),
    verbose = 0
)

# %% [markdown]
# Plot the training history

# %%
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# %% [markdown]
# Now we will try to make recommendation to a random user.

# %%
user_id = df_ratings["User-ID"].sample(1).iloc[0]
books_read_by_user = df_ratings[df_ratings["User-ID"] == user_id]
books_read_by_user

# %%
book_not_read = df_books[~df_books["ISBN"].isin(books_read_by_user["ISBN"])]
book_not_read = list(
    set(book_not_read["ISBN"]).intersection(set(encode_book.keys()))
)
book_not_read[:5]

# %%
book_not_read = [[encode_book.get(x)] for x in book_not_read]
book_not_read[:5]

# %%
user_encode = encode_user.get(user_id)
user_encode

# %%
user_books_array = np.hstack(
    ([[user_encode]] * len(book_not_read), book_not_read)
)
user_books_array

# %%
user_books_array.shape

# %%
ratings = model.predict(user_books_array).flatten()

# %%
top_ratings_indices = ratings.argsort()[-10:][::-1]
top_ratings_indices

# %%
new_books_df = df_books.copy()
new_books_df["id"] = new_books_df["ISBN"].map(encode_book)
new_books_df = new_books_df.dropna()

# %%
top_10_books = new_books_df[new_books_df["id"].isin(top_ratings_indices)][["Book-Title", "Book-Author"]].reset_index(drop=True)
top_10_books


