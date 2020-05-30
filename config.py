import yaml
import os

f = open('config.yaml', 'r')
config = yaml.load(f, Loader=yaml.FullLoader)
f.close()

# for type in config['type']:
#     for task in config['tasks']:
#         if task == 'mnli':
#             command = '''
#             python run_glue_test.py \
#             --mnli True \
#             --data_dir ../bert_embeddings_individual/{}_{} \
#             --output_dir ../simple_classifier_output_individual/{}/{} \
#             --use_individual \
#             --force-overwrite \
#             --num_train_epochs 50 \
#             --n_classes 2 \
#             --verbose
#             '''.format(type.lower(), task, type, task)
#         else:
#             command = '''
#             python run_glue_test.py \
#             --data_dir ../bert_embeddings_individual/{}_{} \
#             --output_dir ../simple_classifier_output_individual/{}/{} \
#             --use_individual \
#             --force-overwrite \
#             --num_train_epochs 50 \
#             --n_classes 2 \
#             --verbose
#             '''.format(type.lower(), task, type, task)
#         os.system(command)
#         print("{} - {}".format(type, task))

tasks = ['qqp', 'mnli', 'snli', 'qnli']
ns = ['2', '3', '3', '2']
for task, n in zip(tasks, ns):
    if task == 'mnli':
        command = '''
        python run_glue_test.py \
        --mnli True \
        --data_dir ../bert_embeddings_individual/qqp_{} \
        --output_dir ../simple_classifier_output_individual/QQP/{} \
        --use_individual \
        --force-overwrite \
        --num_train_epochs 20 \
        --n_classes {} \
        --verbose
        '''.format(task, task, n)
    else:
        command = '''
        python run_glue_test.py \
        --data_dir ../bert_embeddings_individual/qqp_{} \
        --output_dir ../simple_classifier_output_individual/QQP/{} \
        --use_individual \
        --force-overwrite \
        --num_train_epochs 20 \
        --n_classes {} \
        --verbose
        '''.format(task, task, n)
    os.system(command)
    print("{} - {}".format("qqp", task))
