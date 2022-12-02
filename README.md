# Competitions

Create a competition checklist:

- [ ] Create a space using https://huggingface.co/new-space
- [ ] Create a private dataset using https://huggingface.co/new-dataset
- [ ] Create a public dataset using https://huggingface.co/new-dataset
- [ ] Add the following secrets to the space:
  - `AUTOTRAIN_TOKEN`: the token of the user that will be used to create autotrain projects
  - `AUTOTRAIN_USERNAME`: the username of the user that will be used to create autotrain projects
  - `COMPETITION_ID`: Private dataset created previously, e.g.: `my_org/my_private_dataset`

- Private dataset structure:
    
    ```bash
    .
    ├── README.md
    ├── submission_info/
    ├── submissions/
    ├── conf.json
    ├── solution.csv
    ├── COMPETITION_DESC.md
    ├── SUBMISSION_DESC.md

    ````

- Example content for `conf.json`:

    ```
    {
        "SUBMISSION_LIMIT": 5,
        "SELECTION_LIMIT": 2,
        "END_DATE": "2022-11-23",
        "EVAL_HIGHER_IS_BETTER": 1,
        "DATASET": "abhishek/test_competition_dataset",
        "COMPETITION_NAME": "Shoes vs Boots vs Sandals",
        "SUBMISSION_COLUMNS": "id,target",
        "EVAL_METRIC": "accuracy_score"
    }
    ```

- Public dataset structure:
    
    ```bash
    .
    ├── README.md
    ├── training files (folder/zip/csv)
    ├── test files (folder/zip/csv)
    ├── training labels
    ├── sample_submission.csv

    ```
