import bs4
from langchain_community.document_loaders import WebBaseLoader
from GPTme.config import WEB_URLS

def load_webpage(webpage_list=WEB_URLS):

    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(webpage_list),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    return docs