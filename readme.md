#### Document Reading with PyPDF:
It utilizes the `pyPDF` library to read and extract data from `Medical_books` folder. This library efficiently handles large documents, facilitating the structured extraction of valuable medical information.
```python
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
```

#### Data Chunking:
After extracting the data, it is crucial to divide the information into manageable chunks. Chunking helps to:
- Improve retrieval efficiency by enabling quick access to relevant sections.
- Enhance context provided in responses, allowing the model to generate more meaningful answers.
- Mitigate issues related to the maximum token limits of language models by ensuring that only concise and relevant information is processed at any given time.

We also use overlap to keep in the context of these chunks:
```python
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
```

#### Creating Semantic Embeddings:
To facilitate meaningful queries and responses, we employ the HuggingFaceEmbeddings model [(sentence-transformers/all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). 

This model generates semantic embeddings, which capture the context and meaning of the text beyond surface-level words. These embeddings enable the framework to comprehend user queries more effectively and retrieve the most relevant information.
```python
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
```

#### Building a Vector Database with Pinecone:
The next step involves creating a vector database using Pinecone. We load the chunked and embedded data into Pinecone, establishing a knowledge base that can efficiently manage user queries by identifying similar vectors and retrieving pertinent information.
```python
#Initializing index name and the Pinecone
os.environ["PINECONE_API_KEY"] = "Your-Pinecone-API-Key"
index_name="medical-vector"
# Initialize Pinecone with optional parameters
try:
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY"),
        proxy_url=None,            # Example optional parameter
        proxy_headers=None,        # Example optional parameter
        ssl_ca_certs=None,        # Example optional parameter
        ssl_verify=True,  # Example optional parameter, usually set to True
    )
    # Check if the index exists
    indexes = pc.list_indexes()  # List of index names
    index_names = indexes.names()  # Get only the names of the indexes
    if index_name not in index_names:
        print(f'{index_name} does not exist')
        # Change the following line to create the index of your choice
        pc.create_index(
             name=index_name,
             dimension=384,
             metric="cosine",
             spec=ServerlessSpec(
                 cloud="aws",
                 region="us-east-1"
             )
         )
    else:
        print(f'{index_name} exists.')
    # Connect to the existing index
    index = pc.Index(index_name)
except Exception as e:
    print(f"An error occurred while checking indexes: {e}")
# Embedding the text chunks and storing them in Pinecone
try:
    docsearch = LangchainPinecone.from_texts(
        texts=[t.page_content for t in text_chunks],  # Assuming `text_chunks` is a list of text splits
        embedding=embeddings,  # Embedding model instance
        index_name=index_name
    )
except Exception as e:
    print(f"An error occurred while creating embeddings: {e}")
```

