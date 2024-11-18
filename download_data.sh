TARGET_DIR="project_data/downloads"
wget -P "$TARGET_DIR" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -P "$TARGET_DIR" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget -P "$TARGET_DIR" https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip "$TARGET_DIR/owt_train.txt.gz"
wget -P "$TARGET_DIR" https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip "$TARGET_DIR/owt_valid.txt.gz"
