import pandas as pd
from DownloadData import data_file_path

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])


# print(df)
# print(df["Label"].value_counts())

def create_balanced_dataset(df):
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


# Split dataset into 3 parts. These ratios are common in machine learning to train, adjust, and evaluate models.
# Training = 70%
# Validation = 10%
# Testing = 20%
def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate the split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())
# print(balanced_df.shape[0])

# Change the string class labels "ham" and "spam" into integer class labels 0 and 1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
# print(balanced_df)

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
# Test size is implied to be 0.2 as the remainder

print(train_df.shape[0])
print(validation_df.shape[0])
print(test_df.shape[0])

# Save the dataset as CSV (comma-seperated values) files so we can reuse it later
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)