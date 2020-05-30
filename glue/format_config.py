import os

inputs = ["MNLI", 'MRPC', 'Pure', 'QNLI', 'QQP', 'RTE', 'SNLI', 'WNLI']

for name in inputs:
    command = """
    python format_for_glue.py \
    -i ../../simple_classifier_output_individual/{} \
    -o ../../simple_classifier_output_individual/{}/format
    """.format(name, name)
    os.system(command)

for name in inputs:
    command = """
    zip -j -D ../../simple_classifier_output_individual/{}/format/submission.zip ../../simple_classifier_output_individual/{}/format/*.tsv
    """.format(name, name)
    os.system(command)