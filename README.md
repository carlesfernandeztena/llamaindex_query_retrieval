# Document query retrieval using LlamaIndex
Query retrieval over external documents using LlamaIndex.


## Setting the environment
1. Install dependencies in a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Place any documents on the `data` folder.
 
3. Export your Google API key (e.g. for Gemini Pro):
    ```bash
    export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
    ```

## Querying documents
Call the program using 
```bash
python3 -m query_retrieval --query "<YOUR_QUERY_HERE>"
```

This is an example of a query to the [https://arxiv.org/pdf/2312.01479.pdf | OpenVoice paper]
```bash
python3 -m query retrieval --query 
```

returns:
```
1) Flexible Voice Style Control: OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. 
2) Zero-Shot Cross-Lingual Voice Cloning: OpenVoice achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
```

## Additional information

This implementation uses Google's Gemini by default, as it is a powerful LLM for which we can request a free API key. The code also leverages a local embedding model by HugginfFace to avoid incurring unnecessary calls to the LLM. 

You can use additional arguments to control de type of indexing: `summary` indexing, `vector` indexing (the default one), or `keyword` table indexing). Moreover, we can configure the temperature of the LLM, the size of the textual chunks into which the documents are split, or the top-K similarities considered by the query engine. Launch with `--help` to review all options.

```bash
LlamaIndex-based query retrieval on local document files.

options:
  -h, --help            show this help message and exit
  --query QUERY         Query to be used for retrieval.
  --data_folder DATA_FOLDER
                        Directory containing the documents to be parsed.
  --indexing_mode {summary,vector,keyword}
                        Choose the indexing mode: summary, vector, or keyword table.
  --temperature TEMPERATURE
                        Temperature value for the LLM. Range from 0 to 1.
  --chunk_size CHUNK_SIZE
                        Set the chunk size for processing.
  --similarity_top_k SIMILARITY_TOP_K
                        Set the top K similarities to consider.
```

Enjoy! 🎉