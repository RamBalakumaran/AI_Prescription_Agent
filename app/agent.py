import pandas as pd
import re
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import defaultdict

# --- LAZY LOADING IMPLEMENTATION ---
# These global variables will act as a cache. They start as None.
_INTERACTIONS_DF = None
_MEDICINES_DF_STATIC = None
_RAG_CHAIN_CACHE = {} # A dictionary to cache RAG chains

def _load_static_data():
    """This function loads the heavy dataframes, but only when called."""
    global _INTERACTIONS_DF, _MEDICINES_DF_STATIC
    if _MEDICINES_DF_STATIC is None:
        print("--- LAZY LOADING: Loading static CSV data for the first time... ---")
        try:
            _INTERACTIONS_DF = pd.read_csv("data/interactions.csv", encoding='latin1')
            _MEDICINES_DF_STATIC = pd.read_csv("data/medicines.csv", encoding='latin1')
            print("--- Static data loaded successfully. ---")
        except FileNotFoundError as e:
            print(f"FATAL ERROR: Could not load a required CSV file. {e}")
            _INTERACTIONS_DF = pd.DataFrame()
            _MEDICINES_DF_STATIC = pd.DataFrame()

# --- 2. DEFINE THE TOOL FUNCTIONS (Standalone and Reliable) ---

def get_medicine_info(query: str):
    rag_chain = create_rag_chain(search_k=1)
    if rag_chain is None: return "Error: Could not create the information retrieval system."
    result = rag_chain.invoke({"query": query})
    documents = result.get('source_documents')
    if not documents: return "I could not find information for that medicine."
    return format_single_medicine_response(documents[0].page_content, query)

def intelligent_symptom_analyser(query: str):
    print("--- Running Intelligent Symptom Analyser (100% Python Logic) ---")
    _load_static_data() # Ensure data is loaded
    
    medicines_df = _MEDICINES_DF_STATIC.copy()
    medicines_df.fillna('N/A', inplace=True)
    # (The rest of this function is the same...)
    use_case_map = defaultdict(list)
    for index, row in medicines_df.iterrows():
        use_cases = set(term.strip() for term in re.split(r',|\s|/', row['Use_Case'].lower()) if term.strip())
        for use_case in use_cases:
            use_case_map[use_case].append(row['Medicine_Name'])
    query_lower = query.lower().replace('nouse', 'nose').replace('headpain', 'headache').replace('bodypain', 'pain')
    user_symptoms = set(re.findall(r'\b(headache|pain|nose|cough|fever|allergy|inflammation)\b', query_lower))
    if not user_symptoms: return "Please describe your symptoms for an analysis."
    possible_conditions = []
    if 'nose' in user_symptoms or 'cough' in user_symptoms: possible_conditions.append("Common Cold")
    if 'pain' in user_symptoms or 'headache' in user_symptoms or 'fever' in user_symptoms: possible_conditions.append("Flu or Viral Infection")
    if 'allergy' in user_symptoms: possible_conditions.append("Allergic Rhinitis")
    if not possible_conditions: possible_conditions.append("General Symptoms")
    suggested_medicines = {}
    for symptom in user_symptoms:
        for use_case, medicines in use_case_map.items():
            if symptom in use_case:
                for med in medicines:
                    if med not in suggested_medicines:
                        dosage = medicines_df[medicines_df['Medicine_Name'] == med]['Dosage_Instruction'].iloc[0]
                        suggested_medicines[med] = f"• For <strong>{symptom.capitalize()}</strong>, you could consider <strong>{med}</strong> ({dosage})."
    response_parts = [
        f"Based on your symptoms (<strong>{', '.join(user_symptoms)}</strong>), here is a possible analysis:",
        f"<br><strong>🧠 Possible Causes:</strong> {', '.join(possible_conditions)}.",
        "<br><strong>💊 Common Relief Options (from our database):</strong>"
    ]
    if not suggested_medicines: response_parts.append("• No specific medicine suggestions found.")
    else: response_parts.extend(suggested_medicines.values())
    response_parts.extend([
        "<br><strong>⚠️ When to see a doctor:</strong>",
        "• If symptoms persist for more than 3-4 days.",
        "• If you develop a high fever or have difficulty breathing."
    ])
    return "<br>".join(response_parts)

def check_drug_interactions(query: str):
    _load_static_data() # Ensure data is loaded
    mentioned_drugs = [name for name in _MEDICINES_DF_STATIC['Medicine_Name'].unique() if re.search(r'\b' + re.escape(name) + r'\b', query, re.IGNORECASE)]
    if len(mentioned_drugs) < 2: return "Please mention at least two drug names to check for interactions."
    drug1, drug2 = mentioned_drugs[0], mentioned_drugs[1]
    interaction = _INTERACTIONS_DF[
        ((_INTERACTIONS_DF['Medicine_A'].str.lower() == drug1.lower()) & (_INTERACTIONS_DF['Medicine_B'].str.lower() == drug2.lower())) |
        ((_INTERACTIONS_DF['Medicine_A'].str.lower() == drug2.lower()) & (_INTERACTIONS_DF['Medicine_B'].str.lower() == drug1.lower()))
    ]
    if not interaction.empty:
        severity, message = interaction.iloc[0]['Severity'], interaction.iloc[0]['Warning_Message']
        return f"<strong>Interaction Alert ({severity}):</strong><br>{message}"
    else:
        return f"No specific interaction found between <strong>{drug1}</strong> and <strong>{drug2}</strong>.<br><br><strong>Disclaimer:</strong> Always consult a pharmacist."

# --- 3. HELPER FUNCTIONS ---
def create_rag_chain(search_k=1):
    """
    This function now uses a cache. It only builds the heavy RAG chain once.
    """
    global _RAG_CHAIN_CACHE
    if search_k in _RAG_CHAIN_CACHE:
        return _RAG_CHAIN_CACHE[search_k]

    print(f"--- LAZY LOADING: Building RAG chain for k={search_k} for the first time... ---")
    _load_static_data() # Ensure data is loaded
    
    try:
        medicines_df = _MEDICINES_DF_STATIC.copy()
        medicines_df.fillna('N/A', inplace=True)
    except Exception as e:
        return None

    def build_description(row):
        synonyms = row.get('Synonyms', 'N/A')
        return (f"|||Medicine_Name: {row['Medicine_Name']}|||...etc...") # Your full description string
    
    medicines_df['description'] = medicines_df.apply(build_description, axis=1)
    # (The rest of the RAG chain creation logic is the same)
    loader = DataFrameLoader(medicines_df, page_content_column="description")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
    llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2", max_new_tokens=100))
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": search_k}),
        return_source_documents=True
    )
    _RAG_CHAIN_CACHE[search_k] = rag_chain
    print(f"--- RAG chain for k={search_k} cached. ---")
    return rag_chain

# (The parse_context and format_single_medicine_response functions remain the same)
def parse_context(context: str):
    # ... your existing code ...
    pass
def format_single_medicine_response(context: str, original_query: str):
    # ... your existing code ...
    pass

# --- 4. THE ROUTER ---
def get_ai_response(query: str):
    _load_static_data() # Ensure data is loaded before routing
    print(f"Routing query: '{query}'")
    query_lower = query.lower()
    interaction_keywords = ['and', 'with', 'together', 'mix', 'vs', 'versus']
    symptom_keywords = ['symptom', 'feel', 'have', 'suffer', 'pain', 'ache', 'cough', 'headache', 'fever', 'nausea', 'allergy', 'nose', 'nouse', 'knee', 'joint']
    
    mentioned_drugs = set(name for name in _MEDICINES_DF_STATIC['Medicine_Name'].unique() if re.search(r'\b' + re.escape(name) + r'\b', query, re.IGNORECASE))
    
    if len(mentioned_drugs) > 1 or (len(mentioned_drugs) == 1 and any(key in query_lower.split() for key in interaction_keywords)):
        print("--> Routing to: Drug Interaction Checker")
        return check_drug_interactions(query)
    
    if any(key in query_lower for key in symptom_keywords) and len(mentioned_drugs) == 0:
        print("--> Routing to: Intelligent Symptom Analyser")
        return intelligent_symptom_analyser(query)
        
    print("--> Routing to: Medicine Information Finder")
    return get_medicine_info(query)

print("AI Agent is ready (Models will be loaded on first request).")