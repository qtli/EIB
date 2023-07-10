
expl_for_scienceqa="http://ai2-website.s3.amazonaws.com/data/COLING2016_Explanations_Oct2016.zip"
senmaking="https://github.com/wangcunxiang/Sen-Making-and-Explanation/archive/refs/heads/master.zip"
liarplus="https://github.com/Tariq60/LIAR-PLUS/archive/refs/heads/master.zip"
pubhealth="https://drive.google.com/u/0/uc?id=1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj&export=download"
e_delta_nli="https://github.com/fabrahman/RationaleGen/archive/refs/heads/master.zip"

ecqa="https://github.com/dair-iitd/ECQA-Dataset/archive/refs/heads/main.zip"
esnli="https://github.com/OanaMariaCamburu/e-SNLI/archive/refs/heads/master.zip"

echo Download

mkdir -p data/explanation_datasets
cd data/explanation_datasets

wget -c ${expl_for_scienceqa} -O science_qa
unzip science_qa.zip
mv COLING2016_Explanations_Oct2016 science_qa
wget -c ${senmaking} -O senmaking.zip
unzip senmaking.zip
mv Sen-Making-and-Explanation-master senmaking
wget -c ${liarplus} -O liarplus.zip
unzip liarplus.zip
mv LIAR-PLUS-master liarplus
wget -c ${pubhealth} -O pubhealth.zip
unzip pubhealth.zip
mv PUBHEALTH pubhealth
wget -c ${e_delta_nli} -O e_delta_nli.zip
unzip e_delta_nli.zip
mv RationaleGen-master e_delta_nli

wget -c ${ecqa} -O ecqa.zip
unzip ecqa.zip
mv ECQA-Dataset-main ecqa
mkdir -p ecqa/cqa
wget https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl -P ecqa/cqa/
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl -P ecqa/cqa/
wget https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl -P ecqa/cqa/
python ecqa/generate_data.py


wget -c ${esnli} -O esnli.zip
unzip esnli.zip
mv e-SNLI-master esnli


cd ..
mkdir -p utils
wget https://github.com/thu-coai/UNION/raw/master/Data/conceptnet_antonym.txt -P utils/
wget https://github.com/thu-coai/UNION/raw/master/Data/conceptnet_entity.csv -P utils/
wget https://github.com/thu-coai/UNION/raw/master/Data/negation.txt -P utils/


echo unify existing explanation datasets and get explanation_datasets/unify_expl_dataset.json
python unify.py

echo process ...

python retrieve_data.py \
--mode 1 \
--unify_data_file explanation_datasets/unify_expl_dataset.json \
--sentences utils/all_sentences.json


python retrieve_data.py \
--mode 2 \
--model_name_or_path facebook/contriever \
--output_embedding_dir utils/contriever_src/contriever_embeddings  \
--sentences utils/all_sentences.json \
--shard_id 0 \
--num_shards 1


python retrieve_data.py \
--mode 3 \
--model_name_or_path facebook/contriever \
--sentences utils/all_sentences.json \
--sentences_embeddings "utils/contriever_src/contriever_embeddings/*embeddings*" \
--n_sentences 100 \
--output_dir utils/contriever_src/contriever_results \
--save_or_load_index


echo prepare infilling data
wget https://connecthkuhk-my.sharepoint.com/:u:/g/personal/qtli_connect_hku_hk/EaxonMZNvXxBp84iVAI72tYBx8nf14ROBZ-Ra9LpqbSryg?e=A4jUmm\&download=1 -O glm-2b.tar.bz2
mv glm-2b.tar.bz2  utils/infilling/glm
tar -jxvf utils/infilling/glm/glm-2b.tar.bz2


echo process data
python process.py










