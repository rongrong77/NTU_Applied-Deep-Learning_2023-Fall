python ./inference.py \
    --base_model_path "yentinglin/Taiwan-LLM-7B-v2.0-chat" \
    --peft_path "./1129A" \
    --test_data_path "../data/private_test.json" \
    --output_path "./1129.json"