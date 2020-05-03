# simple_classifier
A simple linear classifier for the outputs from project "bert_for_embedding".

Example Usage:
```bash
python simple_classifier/run_classification.py \
    --data_dir "output folder from 'bert_for_embedding'" \
    --mnli True\
    --output_dir "output folder for this task" \
    --num_train_epochs 5 \
    --force-overwrite
``` 