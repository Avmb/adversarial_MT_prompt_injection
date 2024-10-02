#! /bin/sh

#TGT_LANG=$1
API_KEY=$(cat ../oai_key)

rm score_from_openai.jsonl

python ./api_request_parallel_processor.py \
  --requests_filepath score_to_openai.jsonl \
  --save_filepath score_from_openai.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 8000 \
  --max_tokens_per_minute 7000000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20 \
  --api_key $API_KEY

