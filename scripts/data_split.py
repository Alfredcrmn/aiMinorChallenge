"""One reproducible recording-level split, shared by evaluation and EDA."""

from sklearn.model_selection import train_test_split


def split_recordings(data):
    if data.groupby("repetition_id")["exercise"].nunique().ne(1).any():
        raise ValueError("A repetition ID contains multiple exercise labels")
    recordings = data[["repetition_id", "exercise"]].drop_duplicates()
    development, test = train_test_split(
        recordings, test_size=0.2, stratify=recordings["exercise"], random_state=42)
    development_mask = data["repetition_id"].isin(development["repetition_id"])
    test_mask = data["repetition_id"].isin(test["repetition_id"])
    assert not (development_mask & test_mask).any()
    assert (development_mask | test_mask).all()
    return development_mask, test_mask
