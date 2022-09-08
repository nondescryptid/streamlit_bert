import streamlit as st
from pydoc import Doc
from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer,AutoModel, AutoModelForQuestionAnswering
from datasets import load_dataset
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('Answer questions with BERT!')
st.write("This is a simple interface to enter text and ask questions, using functions written by Chris McCormick in his [How to Build Your Own Question Answering System](https://mccormickml.com/2021/05/27/question-answering-system-tf-idf/) article that you should definitely check out if you'd like to learn more about what happens under the hood.")
user_text = [st.text_area("Enter the text you'd like to retrieve information from")]
user_qn = st.text_input("Enter your question")


def answer_question(question, answer_text, tokenizer, model):
    input_ids = tokenizer.encode(question, answer_text, max_length=512,
                                 truncation=True)

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token itself.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]),
                    # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]),
                    # The segment IDs to differentiate question from answer_text
                    return_dict=True)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer 
    
def qna(question,source_text, no_of_answers):
    # Create answers array 
    answers = []
    # Load model 
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Store source_text in variable 
    #  Source text must be in an array: ['text blah blah']
    segmented_source = segment_documents(source_text, 450)
    print(segmented_source)
     
    # Retrieve X most relevant paragraphs to the query
    candidate_docs = get_top_k_articles(question, segmented_source, no_of_answers)

    # Return the likeliest answers from each of our top k most relevant documents in descending order
    print("Here are our top 3 answers")
    for doc in candidate_docs:
      answer = (answer_question(question, doc, tokenizer, model))
      print(doc, "\n END OF DOC")
      answers.append(answer)
      print(answer, "\n")
  
    return answers


# From McCormick
def segment_documents(docs, max_doc_length=450):
  # List containing full and segmented docs
  segmented_docs = []

  for doc in docs:
    # Split document by spaces to obtain a word count that roughly approximates the token count
    split_to_words = doc.split(" ")

    # If the document is longer than our maximum length, split it up into smaller segments and add them to the list 
    if len(split_to_words) > max_doc_length:
      for doc_segment in range(0, len(split_to_words), max_doc_length):
        segmented_docs.append( " ".join(split_to_words[doc_segment:doc_segment + max_doc_length]))

    # If the document is shorter than our maximum length, add it to the list
    else:
      segmented_docs.append(doc)

  return segmented_docs


# From McCormick 
def get_top_k_articles(query, docs, k):
    # Initialize a vectorizer that removes English stop words
    vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

    # Create a corpus of query and documents and convert to TFIDF vectors
    query_and_docs = [query] + docs
    matrix = vectorizer.fit_transform(query_and_docs)

    # Holds our cosine similarity scores
    scores = []

    # The first vector is our query text, so compute the similarity of our query against all document vectors
    for i in range(1, len(query_and_docs)):
        scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

    # Sort list of scores and return the top k highest scoring documents
    sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_doc_indices = [x[0] for x in sorted_list[:k]]
    top_docs = [docs[x] for x in top_doc_indices]
    print("TOP DOCs: ", top_docs)
    return top_docs

if st.button("Let's go"):
  result = qna(user_qn, user_text, 1)
  st.write("Your answer: ", result[0])