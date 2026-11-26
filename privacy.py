import pandas as pd
import numpy as np



def compute_crosstab(df):
    """
    Cross-tabulation: number of people for each (Sex, Age) combination.

    Parameters:
    - df: pandas DataFrame, real or private dataset

    Returns:
    - ct: pandas DataFrame, counts with Sex as rows and age_bin as columns
    """
    ct = pd.crosstab(df["sex"], df["age_bin"])
    # Force all combinations to appear, even with a count of 0
    ct = ct.reindex(index=["Female", "Male"], columns=[0, 1], fill_value=0)
    ct.index.name, ct.columns.name = "sex", "age_bin"
    return ct

def truth_mask(n, eps, rng=None):
    """
    Creates a truth mask of size n for differential privacy. It will return a boolean array where each element is True with probability p = exp(eps) / (exp(eps) + 1) and False with probability 1 - p.
    This is used to determine which entries in a dataset should be considered clean (no noise). 
    """
    if rng is None: #If not none, the truth mask will be the same between each run of the program. (but within a run, several calls will (probably) return different results !) 
        rng = np.random.default_rng()

    p = np.exp(eps) / (np.exp(eps) + 1)
    return rng.random(n) < p

def create_private_dataset(df, eps, rng=None):
    """
    Creates a private dataset by adding noise to the original dataset based on differential privacy.

    Parameters:
    - df: pandas DataFrame, original dataset
    - eps: float, total privacy budget per individual split equally between sex and age_bin
    - (opt) rng: numpy random generator

    Returns:
    - df_private: pandas DataFrame, private dataset
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Create a truth mask for the entire dataset
    sex_mask = truth_mask(len(df), eps/2, rng) # We need eps/2 so it is eps-differentially private for one individual (since we have 2 sensitive attributes, sex and age_bin, we need to split the privacy budget between them).
    age_mask = truth_mask(len(df), eps/2, rng) # We could do eps only but it would become 2*eps-differentially private for one individual.
    
    # Create a copy of the original dataframe to modify
    df_private = df.copy()
    
    # Add noise to the sensitive attributes based on the truth mask
    # Creates inverse array for sex ("Male" becomes "Female" and vice versa)
    sex_flip = np.where(df_private["sex"] == "Male", "Female", "Male")
    df_private["sex"] = np.where(sex_mask, df_private["sex"], sex_flip) # If sex_mask is True, keep the original value, else flip it (take the value of sex_flip).
    # Since age_bin is already a binary feature, we can just flip it by taking 1 - age_bin if the truth mask is False.
    df_private["age_bin"] = np.where(age_mask, df_private["age_bin"], 1 - df_private["age_bin"])
    
    return df_private


def estimate_original_crosstab(private_df, eps):
    """
    Estimate original sex/age counts from a privately randomized dataset.
    
    Parameters:
    - private_df: pandas DataFrame, private dataset
    - eps: float, total privacy budget per individual (before the split between sex and age_bin)
    Returns:
    - estimated_crosstab: pandas DataFrame, estimated counts with Sex as rows and age as columns
    """

    p = 1 / (1 + np.exp(-eps / 2))
    channel = np.array([
        [p, 1 - p],
        [1 - p, p],
    ])

    observed = compute_crosstab(private_df).to_numpy(dtype=float).ravel()
    estimated = np.linalg.solve(np.kron(channel, channel), observed)

    return pd.DataFrame(
        estimated.reshape(2, 2),
        index=["Female", "Male"],
        columns=[0, 1],
    )