## Data Preprocess

### Dev Data Preprocess

1. **Set Up Environment:**
   ```sh
   conda create -n arctic_process_data python=3.9.5
   conda activate arctic_process_data

   apt-get update
   apt-get install -y openjdk-11-jdk

   pip3 install func_timeout ijson tqdm pyserini pyserini==0.22.1 faiss-cpu torch numpy==1.24.3 nltk==3.8.1
   python3 data_preprocessing/nltk_downloader.py
   ```

2. **Generate Evaluation Input Data**

   Here is an example, you should replace with your path

   ```sh
   # Build BM25 index for database values
   python3 data_preprocessing/build_contents_index.py \
   --db-root /BIRD_DIR/dev_20240627/dev_databases \
   --index-root /BIRD_DIR/dev_20240627/db_contents_index \
   --temp-dir /CREATE_TEMP_DIR \
   --threads 16
   ```

   generate input file
   ```
    -i INPUT_JSON       path to input_data_file \
    -o OUTPUT_JSON      path to output_data_file \
    -d DB_PATH          path to the directory of databases \
    -t TABLES_JSON      path to tables definition JSON \
    -s SOURCE           data source name (e.g. "bird") \
    -m MODE             dataset mode (e.g. "dev", "train") \
    -v VALUE_LIMIT_NUM  integer limit for values \
    -c DB_INDEX_PATH    path to db_content_index
   ```
   ```sh
   # Prepare input-output sequences, example here
   bash data_preprocessing/process_dataset.sh \
    -i /BIRD_DIR/dev_20240627/dev.json \
    -o /BIRD_DIR/dev_bird.json \
    -d /BIRD_DIR/dev_20240627/dev_databases/ \
    -t /BIRD_DIR/dev_20240627/dev_tables.json \
    -s bird \
    -m dev \
    -v 2 \
    -c /BIRD_DIR/dev_20240627/db_contents_index
   ```


### Evaluation Reproduction
You can easily reproduce our evaluation results as follows:

1. **Set Up Environment:**
   ```sh
   conda create -n arctic_eval python=3.9.5
   conda activate arctic_eval
   pip3 install vllm func_timeout tqdm matplotlib nltk==3.8.1 sqlparse pandas
   ```

2. **Run Generation and Evaluation:**
Here is an example of evaluation a model, please replace the input paraments
   ```bash
   python3 eval_open_source_models.py \
   --models yangmeng/model_0717 \
   --input_file /BIRD_DIR/dev_bird.json \
   --gold_file_path /BIRD_DIR/dev_20240627/dev.json \
   --dp_path /BIRD_DIR/dev_20240627/dev_databases \
   --self_consistency
   ```



3. **Only Evaluate Generation Results:**
   ```bash
   python3 ./bird_eval/evaluate_bird.py \
   --pred results/model_0720/major_voting_base.json \
   --gold /BIRD_DIR/dev_20240627/dev.json \
   --db_path /BIRD_DIR/dev_20240627/dev_databases \
   --mode major_voting


