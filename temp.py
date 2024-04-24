# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

# %%
df = pd.read_csv("data/racquet.csv")

# %%
df.shape
df

# %% [markdown]
# ### Removing uneccesary coloumns and cleaning up the names

# %%
df.columns


# %%
df = df[['Racquet', 'Power', 'Control', 'Maneuverability', 'Stability', 'Comfort','TouchFeel', 'Topspin', 'Slice', 'Price']]
df.shape

# %%
df = df.rename(columns={"TouchFeel": "Touch"})

# %% [markdown]
# ### Checking and removing duplicates

# %%
df.duplicated(["Racquet"]).sum()
df.drop_duplicates(["Racquet"])

# %% [markdown]
# ### Checking for and removing Null Values

# %%
df.isna().sum() / df.shape[0] 

# %%
df = df.dropna()
df

# %% [markdown]
# ### Plot the data

# %%
# Set Seaborn style
sns.set_style("darkgrid")

# Identify numerical columns
numerical_columns = df.select_dtypes("number").columns

# Plot distribution of each numerical feature
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
	plt.subplot(len(numerical_columns), 2, idx)
	
	# Kernel density plot
	sns.histplot(df[feature], kde=True)
	plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")

# Adjust layout and show plots
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Remove obvious outliers

# %%
df.drop(df[df["Maneuverability"] < 70].index)

# %% [markdown]
# ### Get user input 

# %%
power = input("On a scale of 70-100 how much power do you want in your racket: ")
control = input("On a scale of 70-100 how much control do you want in your racket: ")
maneuverability = input("On a scale of 70-100 how much maneuverability do you want in your racket: ")
stabiity = input("On a scale of 70-100 how much stability do you want in your racket: ")
comfort = input("On a scale of 70-100 how much comfort do you want in your racket: ")
touch = input("On a scale of 70-100 how much touch do you want in your racket: ")
topspin = input("On a scale of 70-100 how much topspin do you want in your racket: ")
slice = input("On a scale of 70-100 how much slice do you want in your racket: ")


# %% [markdown]
# ### Map the racquets to only their specific brand

# %%
brand_list = ["Babolat", "Dunlop", "Head", "Prince", "Wilson", "Yonex", "Tecnifibre", "Volkl"]


df[df['Racquet'].str.contains('|'.join(brand_list), case=False, na=False)]

# Regular expression pattern to extract brand names
pattern = '|'.join(brand_list)

# Extract brand names using str.extract() method
df['Racquet'] = df['Racquet'].str.extract(f'({pattern})', expand=False)

# Drop rows with NaN values after extraction
df = df.dropna()


# %%
df["Racquet"].value_counts()

# %% [markdown]
# ### Extracting Data

# %%
# Extract Data
features = ['Power', 'Control', 'Maneuverability', 'Stability', 'Comfort','Touch', 'Topspin', 'Slice']
X = df[features]
y = df["Racquet"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# %% [markdown]
# ### Hyperparameter Tuning

# %%
# Define a range of k values to test
k_values = list(range(1, 20, 2)) #Odd values from 1 to 20

# Initialize an empty list four our scores
cv_scores = []

# Preform 5-fold cross validation for each k value
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    print(k, scores)
    cv_scores.append(scores.mean())

# %%
# Plot our Cross Validation Results
sns.lineplot(x=k_values, y=cv_scores, marker='o')
plt.title('Accuracy vs. K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)

# %%
numeric = df.select_dtypes("number")
cor = numeric.corr()
sns.heatmap(cor, annot=True)

# %% [markdown]
# ### Feature Selection 

# %%
features = ['Power', 'Control', 'Maneuverability', 'Stability', 'Comfort','Touch', 'Topspin', 'Slice']
y = df["Racquet"]
model = KNeighborsClassifier(n_neighbors=5)
best_score = 0
best_feature = None
selected_features = []
length = len(features)
for i in range(length):
    best_score = 0
    best_feature = None
    for feature in features:
        X = df[[feature] + selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        model = model.fit(X_train, y_train)
        # Evaluate the model
        score = model.score(X_test, y_test) 
        # print("Feature:", feature, "Best Score:", best_score, "Score:", score)
        if score > best_score:
            best_score = score
            best_feature = feature
    selected_features.append(best_feature)
    features.remove(best_feature)
    print("Added: ", best_feature)
    print("Selected Features:", selected_features, "Score:", best_score)

# %% [markdown]
# ## Building Our Model

# %%
df


