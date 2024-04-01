import sys
sys.path.append(".")

import textwrap
from pathlib import Path
from langchain import LLMChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import GCSDirectoryLoader
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from utils.custom_vertex_ai_embeddings import CustomVertexAIEmbeddings
from utils.matching_engine import MatchingEngine
from utils.matching_engine_utils import MatchingEngineUtils
from utils.custom_loaders import custom_load_pdf, custom_load_csv
import vertexai
from vertexai.language_models import TextGenerationModel


import urllib
from pathlib import Path as p
import pandas as pd
import redis
from os import environ as env
from dotenv import load_dotenv
load_dotenv()

r = redis.Redis(
  host=env['HOST'],
  port=15784,
  password=env['PASSWORD'])

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=Path("prompts/qa.prompt").read_text(),
)

eval_prompt = PromptTemplate(
    input_variables=["question", "expected", "answer"],
    template=Path("prompts/eval.prompt").read_text(),
)

namespacekey = 'namespace_list'

class KoraAI:
    #projectid and location make it self variables for all namespaces
    projectid = ""
    location = ""

    def __init__(self, projectid, location):
        self.namespaces = {}
        self.projectid  = projectid
        self.location = location

    def add_namespace(self, bucket, namespace, folder):
        self.namespaces[namespace] = DocqaNamespace(
            bucket, namespace, self.projectid,  self.location, folder)

    def train(self, namespace):
        if namespace in self.namespaces:
            self.namespaces[namespace].train(namespace)
        else:
            raise Exception(
                f"No namespace found with the client name {namespace}")

    def rewrite(self, namespace, question_text):
        if namespace in self.namespaces:
            return self.namespaces[namespace].rewrite(question_text)
        else:
            raise Exception(
                f"No namespace found with the client name {namespace}")
        
    def summarize(self, namespace, url ,type):
        if namespace in self.namespaces:
            return self.namespaces[namespace].summarize(url)
        else:
            raise Exception(
                f"No namespace found with the client name {namespace}")
    
    def generate_text(self, prompt):
        generator = TextGenerator()
        generated_text = generator.generate_long_text(prompt)
        return generated_text

        
    def get_namespaces(self):
        """Retrieve all namespaces from the Redis list.
        
        Returns:
            list: List of namespaces.
        """
        return r.lrange(namespacekey, 0, -1)
    
    def delete_namespace(self, namespace):
        """Delete a specific namespace from the Redis list.
        
        Args:
            namespace (str): The namespace to be deleted.
        
        Returns:
            list: Updated list of namespaces.
        """
        # Retrieve namespaces from Redis and decode them
        namespaces = [r.lindex(namespacekey, i).decode("utf-8") 
                      for i in range(r.llen(namespacekey))]

        # Remove the target namespace if it exists
        if namespace in namespaces:
            namespaces.remove(namespace)

        # Clear the original Redis list
        r.delete(namespacekey)

        # Push the updated list back to Redis
        for item in namespaces:
            r.rpush(namespacekey, item)

        return namespaces
        
    def question(self, namespace, question_text):
        if namespace in self.namespaces:
            return self.namespaces[namespace].question(namespace, question_text)
        else:
            raise Exception(
                f"No namespace found with the client name {namespace}")

    def evaluate_answer(self, namespace, question_text, expected_text, answer_text):
        if namespace in self.namespaces:
            return self.namespaces[namespace].evaluate_answer(question_text, expected_text, answer_text)
        else:
            raise Exception(
                f"No namespace found with the client name {namespace}")


class DocqaNamespace:

    def __init__(self, gcp_bucket, namespace, projectid, region, folder):
        # Constants
        self.PROJECT_ID = projectid
        self.FOLDER = folder
        self.REGION = region
        self.ME_REGION = region
        self.ME_INDEX_NAME = f"{projectid}-me-index"
        self.ME_EMBEDDING_DIR = f"{projectid}-me-embeddings"
        self.ME_DIMENSIONS = 768  # when using Vertex PaLM Embedding
        self.EMBEDDING_QPM = 100
        self.EMBEDDING_NUM_BATCH = 5
        self.NUMBER_OF_RESULTS = 5
        self.SEARCH_DISTANCE_THRESHOLD = 0.6
        self.GCS_BUCKET_DOCS = gcp_bucket
        # Initialization
        vertexai.init(project=self.PROJECT_ID, location=self.REGION)

        self.LLM = VertexAI(
            model_name="text-bison@001",
            max_output_tokens=1024,
            temperature=0,
            top_p=1,
            top_k=1,
            verbose=True,
        )

        self.embeddings = CustomVertexAIEmbeddings(
            requests_per_minute=self.EMBEDDING_QPM,
            num_instances_per_batch=self.EMBEDDING_NUM_BATCH,
        )
        mengine = self.load_matching_engine()
        self.ME_INDEX_ID, self.ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()
        self.me = MatchingEngine.from_components(
            project_id=self.PROJECT_ID,
            region=self.ME_REGION,
            gcs_bucket_name=f'gs://{self.ME_EMBEDDING_DIR}',
            embedding=self.embeddings,
            index_id=self.ME_INDEX_ID,
            endpoint_id=self.ME_INDEX_ENDPOINT_ID,
        )
        r.lpush(namespacekey, namespace)

    def load_matching_engine(self):
        mengine = MatchingEngineUtils(
            self.PROJECT_ID, self.ME_REGION, self.ME_INDEX_NAME)
        index = mengine.create_index(
            embedding_gcs_uri=f"gs://{self.ME_EMBEDDING_DIR}/init_index",
            dimensions=self.ME_DIMENSIONS,
            index_update_method="streaming",
            index_algorithm="tree-ah",
        )
        if index:
            print(index.name)
        index_endpoint = mengine.deploy_index()
        if index_endpoint:
            print(f"Index endpoint resource name: {index_endpoint.name}")
            print(
                f"Index endpoint public domain name: {index_endpoint.public_endpoint_domain_name}")
            print("Deployed indexes on the index endpoint:")
            for d in index_endpoint.deployed_indexes:
                print(f"{d.id}")
        return mengine

    def train(self, namespace):
        print(f"Processing documents from {self.GCS_BUCKET_DOCS}")
        loader = GCSDirectoryLoader(
            project_name=self.PROJECT_ID, bucket=self.GCS_BUCKET_DOCS, prefix=self.FOLDER, loader_func=custom_load_pdf)
        documents = loader.load()
        for document in documents:
            doc_md = document.metadata
            document_name = doc_md["source"].split("/")[-1]
            doc_source_prefix = "/".join(self.GCS_BUCKET_DOCS.split("/")[:3])
            doc_source_suffix = "/".join(doc_md["source"].split("/")[4:-1])
            source = f"{doc_source_prefix}/{doc_source_suffix}"
            document.metadata = {"source": source,
                                 "document_name": document_name, 
                                 "customer_name": namespace}

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        doc_splits = text_splitter.split_documents(documents)
        for idx, split in enumerate(doc_splits):
            split.metadata["chunk"] = idx
        print(f"# of chunks = {len(doc_splits)}")
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [
            [
                {"namespace": "source", "allow_list": [
                    doc.metadata["source"]]},
                {"namespace": "document_name", "allow_list": [
                    doc.metadata["document_name"]]},
                {"namespace": "customer_name", "allow_list": [namespace]},
                {"namespace": "chunk", "allow_list": [
                    str(doc.metadata["chunk"])]},
            ]
            for doc in doc_splits
        ]
        doc_ids = self.me.add_texts(texts=texts, metadatas=metadatas)
        print(doc_ids)

    def fetchqa(self):
        retriever = self.me.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.NUMBER_OF_RESULTS,
                "search_distance": self.SEARCH_DISTANCE_THRESHOLD,
            },
        )
        qa = RetrievalQA.from_chain_type(
            llm=self.LLM,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True,
            chain_type_kwargs={
                "prompt": qa_prompt
            },
        )
        return qa

    def formatter(self, result):
        print(f"Query: {result['query']}")
        print("." * 80)
        if "source_documents" in result.keys():
            for idx, ref in enumerate(result["source_documents"]):
                print("-" * 80)
                print(f"REFERENCE #{idx}")
                print("-" * 80)
                if "score" in ref.metadata:
                    print(f"Matching Score: {ref.metadata['score']}")
                if "source" in ref.metadata:
                    print(f"Document Source: {ref.metadata['source']}")
                if "document_name" in ref.metadata:
                    print(f"Document Name: {ref.metadata['document_name']}")
                print("." * 80)
                print(f"Content: \n{self.wrap(ref.page_content)}")
        print("." * 80)
        print(f"Response: {self.wrap(result['result'])}")
        print("." * 80)

    def wrap(s):
        return "\n".join(textwrap.wrap(s, width=120, break_long_words=False))

    def question(self, namespace, query):
        qa = self.fetchqa()
        qa.retriever.search_kwargs["search_distance"] = self.SEARCH_DISTANCE_THRESHOLD
        qa.retriever.search_kwargs["k"] = self.NUMBER_OF_RESULTS
        qa.retriever.search_kwargs["namespace"] = namespace
        print(query)
        result = qa({"query": query})
        return result

    def rewriteAnswer(self, namespace, question):
        result = self.LLM.generate("Rewrite this answer: ", question)
        return result.generations[0].text

    def evaluateAnswer(self):
        qa = LLMChain(
            llm=self.LLM,
            prompt=eval_prompt)
        return qa

    def evaluate_answer(self, question, expected, answer):
        qa = self.evaluateAnswer()
        print(question)
        print(expected)
        print(answer)
        result = qa({"question": question, "expected": expected, "answer": answer})
        print(result)
        return result
    
    def summarize(self,url):
        """Summarizes a PDF file from a given URL.

           Args:
           url: The URL of the PDF file.

            Returns:
                A Pandas Series containing the concise summaries of each page of the PDF file.
        """
        # download pdf file from url
        data_folder = p.cwd() / "data"
        p(data_folder).mkdir(parents=True, exist_ok=True)
        pdf_url = url
        pdf_file = str(p(data_folder, pdf_url.split("/")[-1]))
        urllib.request.urlretrieve(pdf_url, pdf_file)

        # extract text from pdf file 
        pdf_loader = PyPDFLoader(pdf_file)
        pages = pdf_loader.load_and_split()

        # Create the question prompt template.
        question_prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """
        question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["text"])

        # Create the refine prompt template.
        refine_prompt_template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
              """
        refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["text"])

        # Load the summarize chain.
        refine_chain = load_summarize_chain(self.LLM,chain_type="refine",question_prompt=question_prompt,refine_prompt=refine_prompt,return_intermediate_steps=True,)

        # Generate the refined summaries.
        refine_outputs = refine_chain({"input_documents": pages})

        # Create a Pandas Series containing the refined summaries.
        final_refine_data = []
        for doc, out in zip(
            refine_outputs["input_documents"], refine_outputs["intermediate_steps"]
        ):
            output = {}
            output["file_name"] = p(doc.metadata["source"]).stem
            output["file_type"] = p(doc.metadata["source"]).suffix
            output["page_number"] = doc.metadata["page"]
            output["chunks"] = doc.page_content
            output["concise_summary"] = out
            final_refine_data.append(output)
        
      
        pdf_refine_summary = pd.DataFrame.from_dict(final_refine_data)
        pdf_refine_summary = pdf_refine_summary.sort_values(
            by=["file_name", "page_number"]
            )  # sorting the datafram by filename and page_number
        pdf_refine_summary.reset_index(inplace=True, drop=True)
        pdf_refine_summary.head()


        return pdf_refine_summary['concise_summary']

    
class TextGenerator:
    
    def __init__(self):
        self.model = TextGenerationModel.from_pretrained("text-bison@001")
        self.parameters = {
            "temperature": 0,
            "max_output_tokens": 1024,
            "top_p": 1,
            "top_k": 1
        }

    def generate_long_text(self, prompt):
        # Generate text using the model
        generated_text = self.model.predict(prompt, **self.parameters)
        
        # Format the generated text
        formatted_text = self._format_text(generated_text.text)
        
        return formatted_text
    
    def _format_text(self, text):
        """Format the text by replacing special characters and sequences."""
        text = text.replace('\n', ' ')
        text = text.replace('**', '')
        return text

