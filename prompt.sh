step=$1

if [ "${step}" = "prompting" ]; then
    cd data

    echo download facebook/opt-13b

    mkdir utils/facebook
    mkdir utils/facebook/opt-13b
    wget https://huggingface.co/facebook/opt-13b/resolve/main/.gitattributes -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/LICENSE.md -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/config.json -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/merges.txt -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/pytorch_model-00001-of-00003.bin -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/pytorch_model-00002-of-00003.bin -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/pytorch_model-00003-of-00003.bin -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/pytorch_model.bin.index.json -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/special_tokens_map.json -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/tokenizer_config.json -P utils/facebook/opt-13b/
    wget https://huggingface.co/facebook/opt-13b/resolve/main/vocab.json -P utils/facebook/opt-13b/

    mkdir utils/gpt2
    wget https://huggingface.co/gpt2/resolve/main/.gitattributes -P utils/gpt2/
    wget https://huggingface.co/gpt2/resolve/main/README.md -P utils/gpt2/
    wget https://huggingface.co/gpt2/resolve/main/config.json -P utils/gpt2/
    wget https://huggingface.co/gpt2/resolve/main/generation_config.json -P utils/gpt2/
    wget https://huggingface.co/gpt2/resolve/main/merges.txt -P utils/gpt2/
    wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin -P utils/gpt2/
    wget https://huggingface.co/gpt2/resolve/main/tokenizer.json -P utils/gpt2/
    wget https://huggingface.co/gpt2/resolve/main/vocab.json -P utils/gpt2/

    echo prompt ecqa dataset
    python prompt.py \
    --mode 'ecqa' \
    --test explanation_datasets/ecqa/cqa_data_test.csv \
    --prompt_result explanation_datasets/ecqa/cqa_test_prompt.csv \
    --checkpoint utils/facebook/opt-13b \
    --new_test explanation_datasets/ecqa/cqa_explanation_cands.csv

    echo prompt esnli dataset
    python prompt.py \
    --mode 'esnli' \
    --test explanation_datasets/esnli/dataset/esnli_test.csv \
    --prompt_result explanation_datasets/esnli/esnli_test_prompt.csv \
    --checkpoint utils/facebook/opt-13b \
    --new_test explanation_datasets/esnli/esnli_explanation_cands.csv

elif [ "${step}" = "filtering" ]; then
    echo filter the most prefered explanation candidate

    cd code/Prompting-Filter
    python filter_prompt_result.py

    cd ../..

fi