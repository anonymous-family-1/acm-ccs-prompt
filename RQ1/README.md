# RQ1 Datasets

This directory contains the dataset artifacts used for RQ1.

## Files

### `codechat_manually_judged_prompts.json`

Manual labels for the CodeChat dataset after regex-based filtering.


### `edevgpt_prompt_sensitive_only.json`

Manual labels for the EDevGPT dataset after regex-based filtering, restricted to the sensitive subset kept for this artifact.



### `edev_gpt_unique_prompts_dataset.json`

EDevGPT prompt dataset used as the source corpus.


### `author1.json`
### `author2.json`

`500`-sample annotations files used for Cohen's kappa calculation between the authors

- Records per file: `500`
- Format: JSON array
- Fields:
  - `index`
  - `prompt_text`
  - `category`
  - `is_sensitive`

