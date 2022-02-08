import json
import os

import jsonlines
import torch
import argparse
import random

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from dataset.loader import SciFactJointDataset

from dataset.loader import SciFactJointPredictionData, clean_num, clean_url, clean_invalid_sentence, down_sample
from embedding.jointmodel import JointModelClassifier
from dataset.encode import encode_paragraph
from utils import token_idx_by_sentence, get_rationale_label


class SciFactRetrievalDataset(Dataset):
    def __init__(self, corpus: str, claims: str, sep_token="</s>", k=0, train=True, down_sampling=True):
        # sep_token = ''
        self.rationale_label = {'NOT_ENOUGH_INFO': 0, 'RATIONALE': 1}
        self.rev_rationale_label = {i: l for (l, i) in self.rationale_label.items()}
        # self.abstract_label = {'CONTRADICT': 0, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 2}
        self.abstract_label = {'NOT_ENOUGH_INFO': 0, 'CONTRADICT': 1,  'SUPPORT': 2}
        self.rev_abstract_label = {i: l for (l, i) in self.abstract_label.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        for claim in jsonlines.open(claims):
            all_abstracts = list(corpus)
            if k > 0 and 'doc_ids' in claim:
                candidate_abstract = claim['doc_ids'][:k]
            else:
                candidate_abstract = claim['cited_doc_ids']
            for cand in candidate_abstract:
                all_abstracts.remove(int(cand))
            # print(candidate_abstract)
            candidate_abstract = candidate_abstract + random.sample(all_abstracts, k)
            candidate_abstract = [int(c) for c in candidate_abstract]
            # print(candidate_abstract)
            # print(100*"*")
            if train:
                evidence_doc_ids = [int(id) for id in list(claim['evidence'].keys())]
                cited_doc_ids = [int(id) for id in list(claim['cited_doc_ids'])]
                all_candidates = sorted((list(set(candidate_abstract + evidence_doc_ids + cited_doc_ids))))
            # elif oracle:
            #     if claim['evidence'] != {}:
            #         all_candidates = [int(id) for id in list(claim['evidence'].keys())]
            #     else:
            #         all_candidates = []
            else:
                all_candidates = candidate_abstract
            for doc_id in all_candidates:
                doc = corpus[int(doc_id)]
                # doc_id = str(doc_id)
                abstract_sentences = [sentence.strip() for sentence in doc['abstract']]
                abstract_sentences = clean_invalid_sentence(abstract_sentences)  # #
                if train:
                    if str(doc_id) in claim['evidence']:  # cited_doc is evidence
                        evidence = claim['evidence'][str(doc_id)]
                        # print(str(doc_id), evidence)
                        evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                        # print(evidence_sentence_idx)
                        labels = set(e['label'] for e in evidence)
                        if 'SUPPORT' in labels:
                            label = 'SUPPORT'
                        elif 'CONTRADICT' in labels:
                            label = 'CONTRADICT'
                        else:
                            label = 'NOT_ENOUGH_INFO'

                        if down_sampling:
                            # down samples. Augment the data set, extract sentences from the evidence abstract.
                            kept_sentences, kept_evidence_idx, kept_label = down_sample(abstract_sentences,
                                                                                        evidence_sentence_idx,
                                                                                        label, 0.5, sep_token)
                            if kept_sentences is not None:
                                concat_sentences = (' ' + sep_token + ' ').join(kept_sentences)
                                concat_sentences = clean_num(clean_url(concat_sentences))  # clean the url in the sentence
                                rationale_label_str = ''.join(
                                    ['1' if i in kept_evidence_idx else '0' for i in range(len(kept_sentences))])
                                title = clean_num(clean_url(doc['title']))
                                concat_sentences = title + ' ' + sep_token + ' ' + concat_sentences
                                # concat_sentences = concat_sentences + ' ' + sep_token + ' ' + doc['title']
                                # rationale_label_str = "1" + rationale_label_str

                                self.samples.append({
                                    'claim': claim['claim'],
                                    'claim_id': claim['id'],
                                    'doc_id': doc['doc_id'],
                                    'abstract': concat_sentences,
                                    # 'paragraph': ' '.join(kept_sentences),
                                    'title': ' ' + sep_token + ' '.join(doc['title']),
                                    'sentence_label': rationale_label_str,
                                    'abstract_label': self.abstract_label[kept_label],
                                    'sim_label': 1 if doc['doc_id'] in claim['cited_doc_ids'] else 0,
                                })

                    else:  # cited doc is not evidence
                        evidence_sentence_idx = {}
                        label = 'NOT_ENOUGH_INFO'
                    concat_sentences = (' ' + sep_token +
                                        ' ').join(abstract_sentences)  # concat sentences in the abstract
                    concat_sentences = clean_num(clean_url(concat_sentences))  # clean the url in the sentence
                    rationale_label_str = ''.join(
                        ['1' if i in evidence_sentence_idx else '0' for i in range(len(abstract_sentences))])
                    # print(rationale_label_str)
                    title = clean_num(clean_url(doc['title']))
                    concat_sentences = title + ' ' + sep_token + ' ' + concat_sentences
                    # rationale_label_str = "1" + rationale_label_str

                    self.samples.append({
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'abstract': concat_sentences,
                        # 'paragraph': ' '.join(abstract_sentences),
                        'title': ' ' + sep_token + ' '.join(doc['title']),
                        'sentence_label': rationale_label_str,
                        'abstract_label': self.abstract_label[label],
                        'sim_label': 1 if doc['doc_id'] in claim['cited_doc_ids'] else 0,
                    })
                else:
                    concat_sentences = (' ' + sep_token + ' ').join(abstract_sentences)
                    title = clean_num(clean_url(doc['title']))
                    concat_sentences = title + ' ' + sep_token + ' ' + concat_sentences

                    self.samples.append({
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'abstract': concat_sentences,
                        # 'paragraph': ' '.join(abstract_sentences),
                        'title': ' ' + sep_token + ' '.join(doc['title']),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def schedule_sample_p(epoch, total):
    if epoch == total-1:
        abstract_sample = 1.0
    else:
        abstract_sample = np.tanh(0.5 * np.pi * epoch / (total-1-epoch))
    rationale_sample = np.sin(0.5 * np.pi * epoch / (total-1))
    return abstract_sample, rationale_sample


def retrieval2jsonl(claims_file, retrieval_result):
    claim_ids = []
    claims = {}
    # output_retrieval = jsonlines.open("prediction/abstract_retrieval.jsonl", 'w')
    output_retrieval = "prediction/abstract_retrieval_ARSJoint_test.jsonl"
    assert (len(claims_file) == len(retrieval_result))

    # LABELS = ['NOT_RELATED', 'RELATED']
    for claim, retrieval in zip(claims_file, retrieval_result):
        # retrieval_abstract = []
        claim_id = claim['claim_id']
        claim_ids.append(claim_id)
        curr_claim = claims.get(claim_id, {'id': claim_id, 'claim': claim['claim'], 'doc_ids': []})
        curr_claim['doc_ids'] = curr_claim['doc_ids']
        if retrieval == 1:
            curr_claim['doc_ids'].append(claim['doc_id'])
        claims[claim_id] = curr_claim
    abstract_claim = [claims[claim_id] for claim_id in sorted(list(set(claim_ids)))]
    for i in range(len(abstract_claim)):
        # abstract_claim[i]['doc_ids'] = abstract_claim[i]['doc_ids'][:k]
        abstract_claim[i]['doc_ids'] = abstract_claim[i]['doc_ids']
    # print(abstract_claim[0].keys())
    # print(abstract_claim)
    retrieval_abstract = {}
    with open(output_retrieval, "w") as f:
        for entry in abstract_claim:
            print(json.dumps(entry), file=f)
    # for abstract in abstract_claim:
    #     output_retrieval.write({
    #         'id': abstract['claim_id'],
    #         'claim': abstract['claim'],
    #         'doc_ids': abstract['doc_ids']
    #     })
    return output_retrieval


def get_abstract_retrieval(args, input_set, checkpoint, test=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args).to(device)
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
            abstract_out, rationale_out, retrieval_out = model(encoded_dict, transformation_indices)
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


def evaluation_abstract_retrieval(model, dataset, args, tokenizer):
    model.eval()
    abstract_targets = []
    abstract_outputs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=args.batch_size_gpu, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            abstract_out, rationale_out, retrieval_out = model(encoded_dict, transformation_indices)

            abstract_targets.extend(batch['sim_label'])
            abstract_outputs.extend(retrieval_out)
    return {
            'abstract_micro_f1': f1_score(abstract_targets, abstract_outputs, zero_division=0, average='micro'),
            # 'abstract_f1': tuple(f1_score(abstract_targets, abstract_outputs, zero_division=0, average=None)),
            'abstract_precision': precision_score(abstract_targets, abstract_outputs, zero_division=0, average='micro'),
            'abstract_recall': recall_score(abstract_targets, abstract_outputs, zero_division=0, average='micro'),
        }


def train_model(train_set, dev_set, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args)
    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
        model.reinitialize()
    model = model.to(device)
    parameters = [{'params': model.bert.parameters(), 'lr': args.bert_lr}]
    for module in model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    """
    """
    best_f1 = 0
    best_model = model
    model.train()
    checkpoint = os.path.join(args.save, f'retrievalModel_data.model')
    for epoch in range(args.epochs):
        abstract_sample, rationale_sample = schedule_sample_p(epoch, args.epochs)
        model.train()  # cudnn RNN backward can only be called in training mode
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            # retrieval_out, loss = model(encoded_dict, transformation_indices,
            #                             retrieval_label=batch['sim_label'].to(device))
            # loss.backward()
            _, _, abstract_loss, rationale_loss, sim_loss, bce_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch['abstract_label'].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  retrieval_label=batch['sim_label'].to(device),
                                                                  train=True, rationale_sample=rationale_sample)
            rationale_loss *= args.lambdas[2]
            abstract_loss *= args.lambdas[1]
            sim_loss *= args.lambdas[0]
            bce_loss = args.alpha * bce_loss
            loss = abstract_loss + rationale_loss + sim_loss + bce_loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},')
        scheduler.step()
        train_score = evaluation_abstract_retrieval(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train retrieval score:', train_score)
        dev_score = evaluation_abstract_retrieval(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev retrieval score:', dev_score)
        # save
        # save_path = os.path.join(tmp_dir, str(int(time.time() * 1e5))
        #                          + f'-abstract_f1-{int(dev_score[0]["f1"]*1e4)}'
        #                          + f'-rationale_f1-{int(dev_score[1]["f1"]*1e4)}.model')
        # torch.save(model.state_dict(), save_path)
        if dev_score['abstract_recall'] >= best_f1:
            best_f1 = dev_score['abstract_recall']
            best_model = model
    torch.save(best_model.state_dict(), checkpoint)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    # dataset parameters.
    parser.add_argument('--corpus_path', type=str, default='../data/corpus.jsonl',
                        help='The corpus of documents.')
    parser.add_argument('--claim_train_path', type=str,
                        default='../data/claims_train_retrieved.jsonl')
    parser.add_argument('--claim_dev_path', type=str,
                        default='../data/claims_dev_retrieved.jsonl')
    parser.add_argument('--claim_test_path', type=str,
                        default='../data/claims_dev_retrieved.jsonl')
    parser.add_argument('--gold', type=str, default='../data/claims_dev.jsonl')
    parser.add_argument('--abstract_retrieval', type=str,
                        default='prediction/abstract_retrieval.jsonl')
    parser.add_argument('--rationale_selection', type=str,
                        default='prediction/rationale_selection.jsonl')
    parser.add_argument('--save', type=str, default='model/',
                        help='Folder to save the weights')
    parser.add_argument('--output_label', type=str, default='prediction/label_predictions.jsonl')
    parser.add_argument('--merge_results', type=str, default='prediction/merged_predictions.jsonl')
    parser.add_argument('--output', type=str, default='prediction/result_evaluation.json',
                        help='The predictions.')
    parser.add_argument('--pre_trained_model', type=str)

    # model parameters.
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5, required=False)
    parser.add_argument('--only_rationale', action='store_true')
    parser.add_argument('--batch_size_gpu', type=int, default=8,
                        help='The batch size to send through GPU')
    parser.add_argument('--batch-size-accumulated', type=int, default=256,
                        help='The batch size for each gradient update')
    parser.add_argument('--bert-lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--mode', type=str, default='claim_and_rationale',
                        choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
    parser.add_argument('--filter', type=str, default='structured',
                        choices=['structured', 'unstructured'])

    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension")
    parser.add_argument('--vocab_size', type=int, default=31116)
    parser.add_argument("--dropout", type=float, default=0.5, help="drop rate")
    parser.add_argument('--k', type=int, default=10, help="number of abstract retrieval(training)")
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--lambdas', type=float, default=[1, 2, 12])

    return parser.parse_args()


def printf(args, split):
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    if split:
        print('split: True')
    else:
        print('split: False')
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')


def main():
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args = parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loader dataset
    split = False
    # prediction = False
    if split:
        # split_dataset('../data/claims_train_retrieval.jsonl')
        claim_train_path = '../data/train_data_Bio.jsonl'
        claim_dev_path = '../data/dev_data_Bio.jsonl'
        claim_test_path = '../data/claims_test_retrieved.jsonl'
        # print(claim_test_path)
    else:
        claim_train_path = args.claim_train_path
        claim_dev_path = args.claim_dev_path
        claim_test_path = args.claim_dev_path
    claim_train_path = '../data/claims_train_dev.jsonl'
    claim_dev_path = '../data/claims_train_dev.jsonl'
    claim_test_path = '../data/claims_test_retrieved.jsonl'
    # args.model = 'allenai/scibert_scivocab_cased'
    # args.model = 'model/SciBert_checkpoint'
    # args.pre_trained_model = 'model/pre-train.model'
    args.model = 'dmis-lab/biobert-large-cased-v1.1-mnli'
    # args.model = 't5-small'
    # args.model = 'roberta-large'
    args.epochs = 40
    args.bert_lr = 1e-5
    args.lr = 5e-6
    args.batch_size_gpu = 8
    args.dropout = 0
    args.k = 40
    args.hidden_dim = 1024  # 768/1024
    # args.alpha = 2.3
    args.alpha = 10.5
    # args.lambdas = [11.6, 2.8, 4.1]
    args.lambdas = [9.2, 0.4, 2.3]
    printf(args, split)
    k_train = 6
    print(claim_test_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # print(tokenizer.sep_token)
    train_set = SciFactRetrievalDataset(args.corpus_path, claim_train_path, sep_token=tokenizer.sep_token, k=k_train,
                                        down_sampling=False)
    dev_set = SciFactRetrievalDataset(args.corpus_path, claim_dev_path, sep_token=tokenizer.sep_token, k=k_train,
                                      down_sampling=False)
    # test_set = SciFactJointDataset(args.corpus_path, claim_test_path,
    #                                sep_token=tokenizer.sep_token, k=args.k, down_sampling=False)
    test_set = SciFactJointPredictionData(args.corpus_path, claim_test_path, sep_token=tokenizer.sep_token)
    checkpoint = train_model(train_set, dev_set, args)
    # checkpoint = "model/retrievalModel_data.model"
    retrieval_result = get_abstract_retrieval(args, test_set, checkpoint, test=True)
    retrieval2jsonl(test_set.samples, retrieval_result)


if __name__ == "__main__":
    main()
