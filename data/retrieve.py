import os
import pdb
import json
import pandas
import random
import re
import argparse
import torch
import pickle
import glob
import time
import numpy as np
from tqdm import tqdm

from utils.contriever_src import slurm
from utils.contriever_src import contriever
from utils.contriever_src import utils
from utils.contriever_src import data
from utils.contriever_src import normalize_text
from utils.contriever_src import index
from utils.contriever_src import normalize_text

def transfer_expl_to_expls(expl):
    '''since an explanation could be composed of several valid candidates, it could be very long, so we split it into multiple sub-sentences'''
    expl = expl.strip('\n').strip(' ')
    expls = re.split('\\.|\t', expl)
    new_expl = []
    for e in expls:
        e = e.strip(' ')
        if e == '':
            continue
        if e[-1].isalpha():
            e += ' .'
        new_expl.append(e)
    return new_expl



def get_all_sentences(unify_data, to_file):
    all_sentences = {}

    for data_name in unify_data:
        print('data_name: ', data_name)

        all_sentences[data_name] = []
        if 'explanation_opt' not in unify_data[data_name]['train']:
            continue

        for split in ['train', 'dev', 'test']:
            split_data = unify_data[data_name][split]['explanation_opt'][:50000]
            for idx, expl in enumerate(tqdm(split_data)):
                expl = str(expl)
                if expl.strip() == '' or expl == 'nan':
                    continue

                expls = transfer_expl_to_expls(expl)
                all_sentences[data_name].extend(expls)

    json.dump(all_sentences, open(to_file, 'w'))



def embed_sentences(args, sentences, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, text in enumerate(sentences):
            batch_ids.append(k+1)
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(sentences) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.sentence_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings



def gen_sentence_embeddings(all_sentences_file, args):
    model, tokenizer, _ = contriever.load_retriever(args.model_name_or_path)
    print(f"Model loaded from {args.model_name_or_path}.", flush=True)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    all_sentences = data.load_sentences(all_sentences_file, data_name=None)
    total_sentences = 0
    for k in all_sentences:
        total_sentences += len(all_sentences[k])

    shard_size = total_sentences // args.num_shards
    start_idx = args.shard_id * shard_size  # shard_id=0, num_shards=1 --> 0
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards - 1:  # true
        end_idx = total_sentences

    print(f"Embedding generation for {total_sentences} sentences for all datasets from idx {start_idx} to {end_idx}.")

    data_allids = {}
    data_embeddings = {}
    for data_name in all_sentences:   # 每个数据集单独处理
        if all_sentences[data_name] == []:
            continue
        ids, embeddings = embed_sentences(args, all_sentences[data_name], model, tokenizer)
        data_allids[data_name] = ids
        data_embeddings[data_name] = embeddings

    save_file = os.path.join(args.output_embedding_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(args.output_embedding_dir, exist_ok=True)
    print(f"Saving sentences embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((data_allids, data_embeddings), f)

    print(f"Total sentences processed. Written to {save_file}.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):

    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)

    return embeddings, ids




def index_encoded_data(index, ids, embeddings, indexing_batch_size):
    '''
    split embeddings into chunks not larger than indexing_batch_size, then feed them into index。
    :param index:
    :param ids:
    :param embeddings:
    :param indexing_batch_size: 1000000
    :return:
    '''
    allids = []
    allembeddings = np.array([])

    allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings  # vstack: 矩阵进行列连接
    allids.extend(ids)
    while allembeddings.shape[0] > indexing_batch_size:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:  # false
                q = q.lower()
            if args.normalize_text:  # false
                q = normalize_text.normalize(q)  # Lowercase and remove quotes from a TensorFlow string，比如 text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Query Sentences embeddings shape: {embeddings.size()}")

    return embeddings.numpy()



def add_passages(queries, sentence_id_map, top_sentences_and_scores):
    '''
    add retrieved sentences to original data
    :param queries: 也是sentences
    :param sentence_id_map:
    :param top_passages_and_scores:
    :return:
    '''
    merged_data = {}
    assert len(queries) == len(top_sentences_and_scores)
    for i, d in enumerate(queries):
        results_and_scores = top_sentences_and_scores[i]
        docs = [sentence_id_map[int(doc_id)] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        merged_data[d] = []
        for c in range(ctxs_num):
            if docs[c] != d:
                merged_data[d].append(
                    {
                        "id": results_and_scores[0][c],
                        "sentence": docs[c],
                        "score": scores[c],
                    }
                )
    return merged_data




def sentence_retrieval(all_sentences_file, args):
    '''
    build a dictionary, key is a sentence, value is a list of its neighbour sentences, save this dictionary.
    :param args:
    :return:
    '''

    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    all_sentences = data.load_sentences(all_sentences_file, data_name=None)

    # index all passages
    input_paths = glob.glob(args.sentences_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])  # returns the directory name of the pathname path, e.g., DICTORY/contriever_embeddings

    print(f"Indexing passages from files {input_paths}")
    start_time_indexing = time.time()

    for i, file_path in enumerate(input_paths):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            allids, allembeddings = pickle.load(fin)
            allmerged_data = {}

            for data_name in allids.keys():
                print(f'data_name: {data_name}')
                if all_sentences[data_name] == []:
                    continue
                index = index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)  # projection_size=768, n_subquantizers=0, n_bits=8
                data_embeddings_dir = os.path.join(embeddings_dir, data_name)
                os.makedirs(data_embeddings_dir, exist_ok=True)
                index_path = os.path.join(data_embeddings_dir, "index.faiss")

                if args.save_or_load_index and os.path.exists(index_path):  # save_or_load_index=False
                    index.deserialize_from(data_embeddings_dir)
                else:
                    index_encoded_data(
                        index,
                        allids[data_name],
                        allembeddings[data_name],
                        args.indexing_batch_size
                    )  # indexing_batch_size=1000000

                print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
                if args.save_or_load_index:  # false
                    index.serialize(data_embeddings_dir)

                queries = all_sentences[data_name]
                sentence_id_map = {(idx+1): q for idx, q in enumerate(queries)}
                questions_embedding = embed_queries(args, queries, model, tokenizer)

                # get top k results
                start_time_retrieval = time.time()
                top_ids_and_scores = index.search_knn(questions_embedding, args.n_sentences)
                print(f"Search time: {time.time() - start_time_retrieval:.1f} s.")

                merged_data = add_passages(queries, sentence_id_map, top_ids_and_scores)  # combine query sentences and its neighbours
                allmerged_data[data_name] = merged_data

            output_path = os.path.join(args.output_dir, 'merged_'+os.path.basename(args.sentences))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as fout:
                json.dump(allmerged_data, fout)
            print(f"Saved results to {output_path}")



def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=0, help="mode type")
    parser.add_argument("--unify_data_file", type=str, default='DICTORY/unify_expl_dataset.json', help="Path to unify_data_file (.json file)")
    parser.add_argument("--sentences", type=str, default='all_sentences.json', help="Path to sentences (.json file)")
    parser.add_argument("--sentence_maxlength", type=int, default=512, help="Maximum number of tokens in a sentence")
    parser.add_argument("--prefix", type=str, default="all_sentence_embeddings", help="prefix path to save embeddings")
    parser.add_argument("--output_embedding_dir", type=str, default="contriever_embeddings", help="dir path to save embeddings")
    parser.add_argument("--shard_id", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")

    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of sentences indexed"
    )
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--sentences_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument("--n_sentences", type=int, default=100, help="Number of sentences to retrieve per queries")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    args = parser.parse_args()
    slurm.init_distributed_mode(args)
    return args


if __name__ == '__main__':
    args = parse_params()
    '''
    python retrieve_data.py \
    --unify_data_file explanation_datasets/unify_expl_dataset.json
    --sentences all_sentences.json
    '''
    if args.mode == 1:
        unify_data = json.load(open(args.unify_data_file, 'r'))
        get_all_sentences(
            unify_data=unify_data,
            to_file=args.sentences
        )


    '''
    python retrieve_data.py \
    --model_name_or_path facebook/contriever \
    --output_embedding_dir contriever_embeddings  \
    --sentences all_sentences.json \
    --shard_id 0 \
    --num_shards 1
    '''
    if args.mode == 2:
        gen_sentence_embeddings(
            all_sentences_file=args.sentences,
            args=args
        )

    '''
    python retrieve_data.py \
    --model_name_or_path facebook/contriever \
    --sentences all_sentences.json \
    --sentences_embeddings "contriever_embeddings/*embeddings*" \
    --n_sentences 100 \
    --output_dir contriever_results \
    --save_or_load_index
    '''
    if args.mode == 3:
        sentence_retrieval(
            all_sentences_file=args.sentences,
            args=args
        )
