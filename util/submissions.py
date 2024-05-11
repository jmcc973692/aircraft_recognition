from datetime import datetime

import pandas as pd


def create_submission(predictions, sample_submission_path, submission_save_path):
    # Generate a timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")

    # Modify the submission file path to include the timestamp
    base_path, extension = submission_save_path.rsplit(".", 1)
    timestamped_path = f"{base_path}_{timestamp}.{extension}"

    # Load the sample submission file
    sample_submission = pd.read_csv(sample_submission_path)

    # Convert predictions dict to DataFrame and exclude the last column
    predictions_df = pd.DataFrame.from_dict(predictions, orient="index", columns=sample_submission.columns[1:])

    # Reset index to move filenames to a column
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={"index": "filename"}, inplace=True)

    # Merge the predictions DataFrame with the sample submission DataFrame on 'filename'
    # This ensures that all filenames in the sample submission are included
    # and that their order is preserved
    final_submission = sample_submission[["filename"]].merge(predictions_df, on="filename", how="left")

    # Fill NaN values if there are filenames in the sample submission without predictions
    final_submission.fillna(0, inplace=True)  # Assuming a 0 probability where predictions are missing

    # Save the submission file
    final_submission.to_csv(timestamped_path, index=False)
    print(f"Submission created: {timestamped_path}")


if __name__ == "__main__":
    # Example call to function with debugging
    predictions = {
        "img_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "img_7": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    }  # Adjust example as needed
    create_submission(predictions, "input/sample_submission_copy.csv", "submissions/submission.csv")
