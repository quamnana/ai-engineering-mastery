from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from newspaper import Article
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain

load_dotenv()


def llm_selection(model_type, model_name, temperature=0):
    llm = None
    try:
        if model_type == "openai":
            llm = ChatOpenAI(model=model_name, temperature=temperature)
        elif model_type == "ollama":
            llm = ChatOllama(model=model_name, timeout=120, temperature=temperature)
        else:
            raise ValueError("invalid model type")

        return llm
    except Exception as e:
        print("Model Selection Error: ", e)


def get_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article
    except Exception as e:
        raise ValueError("Error getting articles: ", e)


def preprocess_article(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # split article text into chunks
    chunks = text_splitter.split_text(text)

    # convert text chunks into langchain documents
    documents = [Document(page_content=c) for c in chunks]

    return documents


def get_prompts(summary_type="detailed"):
    # Define prompts based on summary type
    if summary_type == "detailed":
        map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""

        combine_prompt_template = """Write a detailed summary of the following text that combines the previous summaries:
            "{text}"
            FINAL DETAILED SUMMARY:"""
    else:  # concise summary
        map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""

        combine_prompt_template = """Write a concise summary of the following text that combines the previous summaries:
            "{text}"
            FINAL CONCISE SUMMARY:"""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    return map_prompt, combine_prompt


def summarize_article(llm, map_prompt, combine_prompt, documents):
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True,
    )

    summary = chain.invoke(documents)
    return summary


def main():

    # url = (
    #     "https://www.cnn.com/2026/01/11/americas/nobel-institute-prize-transfer-machado"
    # )

    url = "https://sloanreview.mit.edu/article/how-llms-work/"
    # model_type = "openai"
    # model_name = "gpt-4o-mini"

    model_type = "ollama"
    model_name = "llama3.2"
    summary_type = "concise"

    llm = llm_selection(model_type=model_type, model_name=model_name)
    print(llm)

    map_prompt, combine_prompt = get_prompts(summary_type=summary_type)

    article = get_article(url)

    documents = preprocess_article(article.text)

    summary = summarize_article(
        llm=llm,
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        documents=documents,
    )

    response = {
        "title": article.title,
        "text": article.text,
        "authors": article.authors,
        "publish_date": article.publish_date,
        "summary": summary,
        "url": url,
        "model_info": {"type": model_type, "name": model_name},
    }

    print(response)


if __name__ == "__main__":
    main()
