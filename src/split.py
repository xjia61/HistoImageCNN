from sklearn.model_selection import GroupShuffleSplit


def split_by_patient(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split dataframe by patient_id.

    Important:
    Images from the same patient will only appear in one split:
    train, val, or test.
    """

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    # Step 1: train vs temp
    temp_size = val_size + test_size

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=temp_size,
        random_state=random_state
    )

    train_idx, temp_idx = next(
        gss.split(
            df,
            y=df["label"],
            groups=df["patient_id"]
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Step 2: val vs test
    relative_test_size = test_size / temp_size

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=random_state
    )

    val_idx, test_idx = next(
        gss2.split(
            temp_df,
            y=temp_df["label"],
            groups=temp_df["patient_id"]
        )
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def check_patient_overlap(train_df, val_df, test_df):
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    print("Train-Val overlap:", train_patients & val_patients)
    print("Train-Test overlap:", train_patients & test_patients)
    print("Val-Test overlap:", val_patients & test_patients)

    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0

    print("Patient-level split check passed.")