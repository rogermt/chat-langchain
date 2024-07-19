from typing import Dict, Any, Optional
import logging
import os
import re
import subprocess
import traceback
from parser import langchain_docs_extractor

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from constants import WEAVIATE_DOCS_INDEX_NAME, PINECONE_DOCS_INDEX_NAME, PINECONE, WEAVIATE
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX

from langchain_weaviate import WeaviateVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device() -> str:
    try:
        # Check for NVIDIA GPU
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if nvidia_smi.returncode == 0:
            return "cuda"
        # Check for AMD GPU
        rocm_smi = subprocess.run(
            ["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if rocm_smi.returncode == 0:
            return "rocm"
    except FileNotFoundError:
        pass
    # If no GPU detected or commands not found, return "cpu"
    return "cpu"


def get_embeddings_model_openai() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


def get_embeddings_model_huggingface() -> HuggingFaceEmbeddings:
    device = get_device()
    embedding_model_name = "thenlper/gte-base"
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 100},
    )
    return embedding_model


def get_embeddings_model() -> Embeddings:
    return get_embeddings_model_huggingface()


def metadata_extractor(
    meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None
) -> dict:
    title_element = soup.find("title")
    description_element = soup.find("meta", attrs={"name": "description"})
    html_element = soup.find("html")
    title = title_element.get_text() if title_element else ""
    if title_suffix is not None:
        title += title_suffix

    return {
        "source": meta["loc"],
        "title": title,
        "description": description_element.get("content", "")
        if description_element
        else "",
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=False,
        timeout=1200,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str | BeautifulSoup) -> str:
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    elif isinstance(html, BeautifulSoup):
        soup = html
    else:
        raise ValueError(
            "Input should be either BeautifulSoup object or an HTML string"
        )
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=False,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def load_langgraph_docs():
    return SitemapLoader(
        "https://langchain-ai.github.io/langgraph/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={"parse_only": SoupStrainer(name=("article", "title"))},
        meta_function=lambda meta, soup: metadata_extractor(
            meta, soup, title_suffix=" | 🦜🕸️LangGraph"
        ),
    ).load()


def ingest_docs_weaviate(embedding):
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    
    client = weaviate.connect_to_wcs(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    )
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        attributes=["source", "title"],
    )

    namespace = f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}"  
    return client, vectorstore, namespace


def ingest_docs_pinecone(embedding):
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    client = Pinecone(
        api_key=PINECONE_API_KEY
    )
    spec = ServerlessSpec(
                cloud=os.environ["PINECONE_CLOUD"],
                region=os.environ["PINECONE_REGION"])

    # First, check if our index already exists. If it doesn't, we create it
    if PINECONE_DOCS_INDEX_NAME not in client.list_indexes().names():
        PINECONE_INDEX_DIMENSION = 768
        # we create a new index
        client.create_index(
          name=PINECONE_DOCS_INDEX_NAME,
          metric='cosine',
          dimension=PINECONE_INDEX_DIMENSION,
          spec=spec
    )
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_DOCS_INDEX_NAME, 
        embedding=embedding
    )
    namespace =  f"pinecone/{PINECONE_DOCS_INDEX_NAME}"
    return client, vectorstore, namespace

    
def do_upsert(
    docs_transformed,
    record_manager,
    vectorstore,
    cleanup: str = "full",
    source_id_key: str = "source",
    force_update: bool = False
) -> Dict[str, Any]:
    try:
        force_update = (os.environ.get("FORCE_UPDATE") or "false").lower() == "true"
        
        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup=cleanup,
            source_id_key=source_id_key,
            force_update=force_update,
        )
        return indexing_stats
    except Exception as e:
        error_message = f"An error occurred during indexing: {str(e)}"
        print(error_message)
        print("Traceback:")
        traceback.print_exc()
        
        # You might want to return some information about the error
        return {
            "error": error_message,
            "traceback": traceback.format_exc()
        }
    

def ingest_docs(vectordb=WEAVIATE):
     
    vectordb_functions = {
        "WEAVIATE": ingest_docs_weaviate,
        "PINECONE": ingest_docs_pinecone
    }
    record_manager_db_url = os.environ["RECORD_MANAGER_DB_URL"]
    vectordb = os.environ.get('VECTOR_DB', vectordb)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()
   
    if vectordb in vectordb_functions:
        ingest_function = vectordb_functions[vectordb]
        client, vectorstore, namespace = ingest_function(embedding)
    else:
        print(f"Unknown vectordb value: {vectordb}")

    record_manager = SQLRecordManager(
        namespace, db_url=record_manager_db_url
    )
    record_manager.create_schema()

    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")
    docs_from_langgraph = load_langgraph_docs()
    logger.info(f"Loaded {len(docs_from_langgraph)} docs from LangGraph")


    docs_transformed = text_splitter.split_documents(
        docs_from_documentation
        + docs_from_api
        + docs_from_langsmith
        + docs_from_langgraph
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = do_upsert(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    if vectordb == WEAVIATE:
        num_vecs = (
            client.collections.get(WEAVIATE_DOCS_INDEX_NAME)
            .aggregate.over_all()
            .total_count
        )
    elif vectordb == PINECONE:
        default_namespace = ""
        index_stats = client.Index(PINECONE_DOCS_INDEX_NAME).describe_index_stats()
        num_vecs = index_stats["namespaces"][default_namespace]["vector_count"]
    else:
        num_vecs = 0
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )



if __name__ == "__main__":
    ingest_docs()