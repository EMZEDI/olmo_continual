import os
import pandas as pd
import argparse
import numpy as np

def const_flip(df, flip_ratio):
    """Randomly flip the preference examples in the dataset

    Args:
        df (pandas.DataFrame): The dataframe of preference examples
        flip_ratio (float): The ratio of examples to flip
    """
    # Randomly flip the preference examples
    flip_indices = np.random.choice(df.index, int(flip_ratio * len(df)), replace=False)
    df.loc[flip_indices, ['chosen', 'rejected']] = df.loc[flip_indices, ['rejected', 'chosen']].values

    return df

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Clean the preference dataset')
    parser.add_argument('--const_number', type=int, default=1, help='The number of the constitution to clean')
    args = parser.parse_args()
    const_number = args.const_number

    direct = f"data/preferences/{const_number}"
    all_files = os.listdir(direct)
    df = pd.concat([pd.read_csv(os.path.join(direct, file)) for file in all_files])
    df.info()
    # df = df.head()  # Uncomment for testing with a small dataset

    # Create a new DataFrame for the preference examples
    preference_examples = pd.DataFrame()
    preference_examples['prompt'] = df['prompt']
    preference_examples['chosen'] = df['response_1']
    preference_examples['rejected'] = df['response_2']

    if const_number == 2:   # just for testing purposes
        preference_examples = const_flip(preference_examples, 0.6)

    # Save the new dataset
    preference_examples.to_csv(f"data/cleaned_preference_dataset_constitution_{const_number}.csv", index=False)
    print(preference_examples.head())
    print(f"Dataset created and saved to 'cleaned_preference_dataset_constitution_{const_number}.csv'")