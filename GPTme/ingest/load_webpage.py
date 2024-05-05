import bs4
from langchain_community.document_loaders import WebBaseLoader
from GPTme.config import WEB_URLS

def load_webpage(webpage_list=WEB_URLS):
    """
    This function loads webpages from a list of URLs and returns a list of BeautifulSoup objects.

    Parameters:
    webpage_list (list): A list of URLs to load.

    Returns:
    docs (list): A list of BeautifulSoup objects containing the requested HTML.

    """
    try:
        # Only keep post title, headers, and content from the full HTML.
        bs4_strainer = bs4.SoupStrainer(class_=(
            "post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=(webpage_list),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()

        return docs
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
