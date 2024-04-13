from datetime import datetime

import pandas as pd


def create_submission(predictions, sample_submission_path, submission_save_path):
    # Generate a timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")

    # Modify the submission file path to include the timestamp
    base_path, extension = submission_save_path.rsplit(".", 1)
    timestamped_path = f"{base_path}_{timestamp}.{extension}"

    sample_submission = pd.read_csv(sample_submission_path)
    filenames = sample_submission["filename"].tolist()

    formatted_predictions = [[filename] + list(prediction) for filename, prediction in zip(filenames, predictions)]

    assert len(formatted_predictions) == len(
        sample_submission
    ), "Mismatch in rows between formatted predictions and sample submission"

    submission = pd.DataFrame(formatted_predictions, columns=sample_submission.columns)
    submission.to_csv(timestamped_path, index=False)

    print(f"Submission created: {timestamped_path}")
