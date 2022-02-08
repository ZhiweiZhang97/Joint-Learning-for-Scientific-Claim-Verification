import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from embedding.jointmodel import JointModelClassifier
from embedding.retrieval import JointModelRetrieval
from evaluation.evaluation_model import evaluation_joint, evaluation_abstract_retrieval
from dataset.encode import encode_paragraph
from dataset.loader import SciFactJointDataset
from utils import token_idx_by_sentence, get_rationale_label
from sklearn.metrics import f1_score, precision_score, recall_score


def get_abstract_retrieval(args, input_set, checkpoint, test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelRetrieval(args).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    retrieval_result = []
    abstract_targets = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(input_set, batch_size=1, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            retrieval_out = model(encoded_dict, transformation_indices)
            retrieval_result.extend(retrieval_out)
            if not test:
                abstract_targets.extend(batch['sim_label'])
    if not test:
        score = {
            'abstract_micro_f1': f1_score(abstract_targets, retrieval_result, zero_division=0, average='micro'),
            'abstract_precision': precision_score(abstract_targets, retrieval_result, zero_division=0, average='micro'),
            'abstract_recall': recall_score(abstract_targets, retrieval_result, zero_division=0, average='micro'),
        }
        print(f'prediction retrieval score:', score)

    return retrieval_result


def teacher_parse_args():
    parse = argparse.ArgumentParser(
        'args for teacher model'
    )
    parse.add_argument('--model', type=str, default='')
    parse.add_argument('--dropout', type=float, default=0)
    parse.add_argument('--hidden_dim', type=int, default=1024)
    return parse.parse_args()


def student_parse_args():
    parse = argparse.ArgumentParser(
        'args for student model'
    )
    parse.add_argument('--model', type=str, default='')
    parse.add_argument('--dropout', type=float, default=0)
    parse.add_argument('--hidden_dim', type=int, default=768)
    return parse.parse_args()


def schedule_sample_p(epoch, total):
    if epoch == total-1:
        abstract_sample = 1.0
    else:
        abstract_sample = np.tanh(0.5 * np.pi * epoch / (total-1-epoch))
    rationale_sample = np.sin(0.5 * np.pi * epoch / (total-1))
    return abstract_sample, rationale_sample


def main():
    teacher_args = teacher_parse_args()
    student_args = student_parse_args()
    teacher_args.model = 'dmis-lab/biobert-large-cased-v1.1-mnli'
    student_args.model = 'dmis-lab/biobert-base-cased-v1.1-mnli'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # loader dataset
    k_train = 12
    claim_train_path = '../data/claims_train_retrieved.jsonl'
    claim_dev_path = '../data/claims_dev_retrieved.jsonl'
    corpus_path = '../data/corpus.jsonl'
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_args.model)
    student_tokenizer = AutoTokenizer.from_pretrained(student_args.model)
    # print(teacher_tokenizer)
    # print(student_tokenizer)
    train_set = SciFactJointDataset(corpus_path, claim_train_path, sep_token=teacher_tokenizer.sep_token, k=k_train)
    dev_set = SciFactJointDataset(corpus_path, claim_dev_path, sep_token=teacher_tokenizer.sep_token, k=k_train,
                                  down_sampling=False)

    # loader teacher model
    checkpoint = 'model/retrievalModel_data.model'
    teacher_model = JointModelClassifier(teacher_args).to(device)
    teacher_model.load_state_dict(torch.load(checkpoint))
    teacher_model = teacher_model.to(device)
    # print(teacher_model)

    # train student model
    epochs = 40
    student_model = JointModelRetrieval(student_args)
    student_model = student_model.to(device)
    print(next(student_model.parameters()).device)
    parameters = [{'params': student_model.bert.parameters(), 'lr': 1e-5},
                  {'params': student_model.abstract_retrieval.parameters(), 'lr': 5e-6}]
    for module in student_model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': 1e-5})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs)

    # loss_fun = nn.CrossEntropyLoss()
    criterion = nn.KLDivLoss()
    best_student_model = student_model
    min_loss = 9999999999999
    student_checkpoint = 'model/distillation_model.model'

    for epoch in range(epochs):
        student_model.train()
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        abstract_sample, rationale_sample = schedule_sample_p(epoch, epochs)
        for i, batch in enumerate(t):
            teacher_encoded_dict = encode_paragraph(teacher_tokenizer, batch['claim'], batch['abstract'])
            student_encoded_dict = encode_paragraph(student_tokenizer, batch['claim'], batch['abstract'])
            teacher_transformation_indices = token_idx_by_sentence(teacher_encoded_dict['input_ids'], teacher_tokenizer.sep_token_id,
                                                           student_args.model)
            student_transformation_indices = token_idx_by_sentence(student_encoded_dict['input_ids'], student_tokenizer.sep_token_id,
                                                                    teacher_args.model)
            teacher_encoded_dict = {key: tensor.to(device) for key, tensor in teacher_encoded_dict.items()}
            student_encoded_dict = {key: tensor.to(device) for key, tensor in student_encoded_dict.items()}
            teacher_transformation_indices = [tensor.to(device) for tensor in teacher_transformation_indices]
            student_transformation_indices = [tensor.to(device) for tensor in student_transformation_indices]
            # print(list(student_transformation_indices))
            # print(list(teacher_transformation_indices))
            student_retrieval_out, student_retrieval_loss = student_model(student_encoded_dict, student_transformation_indices,
                                                                          retrieval_label=batch['sim_label'].to(device))
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            teacher_retrieval_out, teacher_retrieval_loss = teacher_model(teacher_encoded_dict, teacher_transformation_indices,
                                                                                         abstract_label=batch['abstract_label'].to(device),
                                                                                         rationale_label=padded_label.to(device),
                                                                                         retrieval_label=batch['sim_label'].to(device),
                                                                                         train=True, rationale_sample=rationale_sample,
                                                                                         retrieval_state = True)

            # loss between student model output and true label: student_retrieval_loss
            # KLDivLoss between student model and teacher model
            # print(student_retrieval_out)
            # print(100 * '*')
            # print(teacher_retrieval_out)
            KL_loss = criterion(student_retrieval_out.log(), teacher_retrieval_out)
            loss = KL_loss + student_retrieval_loss
            t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < min_loss:
                best_student_model = student_model
                min_loss = loss
        scheduler.step()
    torch.save(best_student_model.state_dict(), student_checkpoint)
    retrieval_result = get_abstract_retrieval(student_args, dev_set, student_checkpoint)
    # torch.save(student_model.state_dict(), student_checkpoint)


if __name__ == "__main__":
    main()

