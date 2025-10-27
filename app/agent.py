# app/agent.py

import pandas as pd
import re
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- CORRECTED IMPORTS: Adjusted path for create_retrieval_chain ---
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# --- END CORRECTION ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import defaultdict

# --- LAZY LOADING IMPLEMENTATION ---
_INTERACTIONS_DF = None
_MEDICINES_DF_STATIC = None
_RAG_CHAIN_CACHE = {}

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
    
    result = rag_chain.invoke({"input": query})
    documents = result.get('context')

    if not documents: return "I could not find information for that medicine."
    return format_single_medicine_response(documents[0].page_content, query)

def intelligent_symptom_analyser(query: str):
    # This function is unchanged
    print("--- Running Intelligent Symptom Analyser (Context-Aware) ---")
    _load_static_data()
    medicines_df = _MEDICINES_DF_STATIC.copy()
    medicines_df.fillna('N/A', inplace=True)
    use_case_map = defaultdict(list)
    for index, row in medicines_df.iterrows():
        use_cases = set(term.strip() for term in re.split(r',|\s|/', row['Use_Case'].lower()) if term.strip())
        for use_case in use_cases:
            use_case_map[use_case].append(row['Medicine_Name'])
    query_lower = query.lower().replace('nouse', 'nose').replace('pain', ' pain ')
    symptom_categories = {'systemic':['fever','headache','fatigue','chills'],'respiratory':['cough','nose','congestion','throat'],'musculoskeletal':['knee','joint','sprain','muscle','body','inflammation'],'digestive':['acidity','nausea','vomiting','diarrhea']}
    detected_symptoms = defaultdict(list)
    match = re.search(r'\b(knee|joint|muscle|stomach|head)\s*pain\b', query_lower)
    if match:
        pain_type = match.group(1)
        detected_symptoms['musculoskeletal' if pain_type != 'head' else 'systemic'].append(f"{pain_type} pain")
    for category, keywords in symptom_categories.items():
        for keyword in keywords:
            if keyword in query_lower and keyword not in str(detected_symptoms):
                detected_symptoms[category].append(keyword)
    if not detected_symptoms: return "Please describe your symptoms more clearly (e.g., 'I have a headache and a cough') for an analysis."
    possible_conditions = []
    if 'systemic' in detected_symptoms or 'respiratory' in detected_symptoms: possible_conditions.append("Common Cold or Flu")
    if 'musculoskeletal' in detected_symptoms and not possible_conditions: possible_conditions.append("Musculoskeletal Strain or Localized Inflammation")
    if 'digestive' in detected_symptoms: possible_conditions.append("Digestive Discomfort")
    if not possible_conditions: possible_conditions.append("General Symptoms")
    suggested_medicines = {}
    all_detected_keywords = [item for sublist in detected_symptoms.values() for item in sublist]
    for keyword in all_detected_keywords:
        search_term = keyword.split(' ')[-1]
        if search_term in use_case_map:
            med_name = use_case_map[search_term][0]
            if med_name not in suggested_medicines:
                dosage = medicines_df[medicines_df['Medicine_Name'] == med_name]['Dosage_Instruction'].iloc[0]
                suggested_medicines[med_name] = f"• For <strong>{keyword.capitalize()}</strong>, you could consider <strong>{med_name}</strong> ({dosage})."
    user_symptoms_str = ', '.join(all_detected_keywords)
    response_parts = [f"Based on your symptoms (<strong>{user_symptoms_str}</strong>), here is a possible analysis:",f"<br><strong>🧠 Possible Causes:</strong> {', '.join(possible_conditions)}.", "<br><strong>💊 Common Relief Options ):</strong>"]
    if not suggested_medicines: response_parts.append("• No specific medicine suggestions found for these exact symptoms.")
    else: response_parts.extend(suggested_medicines.values())
    response_parts.extend(["<br><strong>⚠️ When to see a doctor:</strong>", "• If symptoms persist for more than 3-4 days.","• If you develop a high fever or have difficulty breathing.","• If the pain is severe and does not improve."])
    return "<br>".join(response_parts)

def check_drug_interactions(query: str):
    # This function is unchanged
    _load_static_data()
    mentioned_drugs = [name for name in _MEDICINES_DF_STATIC['Medicine_Name'].unique() if re.search(r'\b' + re.escape(name) + r'\b', query, re.IGNORECASE)]
    if len(mentioned_drugs) < 2: return "Please mention at least two drug names to check for interactions."
    drug1, drug2 = mentioned_drugs[0], mentioned_drugs[1]
    interaction = _INTERACTIONS_DF[((_INTERACTIONS_DF['Medicine_A'].str.lower() == drug1.lower()) & (_INTERACTIONS_DF['Medicine_B'].str.lower() == drug2.lower())) | ((_INTERACTIONS_DF['Medicine_A'].str.lower() == drug2.lower()) & (_INTERACTIONS_DF['Medicine_B'].str.lower() == drug1.lower()))]
    if not interaction.empty:
        severity, message = interaction.iloc[0]['Severity'], interaction.iloc[0]['Warning_Message']
        return f"<strong>Interaction Alert ({severity}):</strong><br>{message}"
    else:
        return f"No specific interaction found between <strong>{drug1}</strong> and <strong>{drug2}</strong>.<br><br><strong>Disclaimer:</strong> Always consult a pharmacist."

# --- 3. HELPER FUNCTIONS ---
def create_rag_chain(search_k=1):
    global _RAG_CHAIN_CACHE
    if search_k in _RAG_CHAIN_CACHE:
        return _RAG_CHAIN_CACHE[search_k]
    print(f"--- LAZY LOADING: Building RAG chain for k={search_k} for the first time... ---")
    _load_static_data()
    try:
        medicines_df = _MEDICINES_DF_STATIC.copy()
        medicines_df.fillna('N/A', inplace=True)
    except Exception as e:
        return None
    def build_description(row):
        synonyms = row.get('Synonyms', 'N/A')
        return (f"|||Medicine_Name: {row['Medicine_Name']}|||Strength: {row['Strength']}|||Use Case: {row['Use_Case']}|||"
                f"Alternative: {row['Alternative']}|||Stock: {row['Stock']}|||Dosage: {row['Dosage_Instruction']}|||Synonyms: {synonyms}|||")
    medicines_df['description'] = medicines_df.apply(build_description, axis=1)
    loader = DataFrameLoader(medicines_df, page_content_column="description")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
    llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2", max_new_tokens=100))
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based only on the provided context:\n\n"
        "<context>\n{context}\n</context>\n\n"
        "Question: {input}"
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    _RAG_CHAIN_CACHE[search_k] = rag_chain
    print(f"--- RAG chain for k={search_k} cached. ---")
    return rag_chain

def parse_context(context: str):
    # This function is unchanged
    data = {}
    parts = [p.strip() for p in context.split('|||') if p.strip()]
    for part in parts:
        try:
            key, value = part.split(':', 1)
            data[key.strip()] = value.strip()
        except ValueError: continue
    return data

def format_single_medicine_response(context: str, original_query: str):
    # This function is unchanged
    data = parse_context(context)
    if 'Medicine_Name' not in data: return "Found incomplete information."
    db_medicine_name, strength = data.get('Medicine_Name'), data.get('Strength')
    synonyms = [s.strip() for s in data.get('Synonyms', '').split(',') if s.strip()]
    user_term = next((s for s in synonyms if re.search(r'\b' + re.escape(s) + r'\b', original_query, re.IGNORECASE)), None)
    title = f"Here is the information for <strong>{db_medicine_name} {strength}</strong>"
    title += f" (also known as <strong>{user_term}</strong>):" if user_term and user_term.lower() != db_medicine_name.lower() else ":"
    response_parts = [title,f"•  <strong>Use Case:</strong> {data.get('Use Case', 'N/A')}",f"•  <strong>Availability:</strong> {'In Stock' if data.get('Stock', 'No').lower() == 'yes' else 'Out of Stock'}",f"•  <strong>Alternative:</strong> {data.get('Alternative', 'N/A')}",f"•  <strong>Dosage:</strong> {data.get('Dosage', 'N/A')}", "<br><strong>Disclaimer:</strong> Always consult a doctor or pharmacist."]
    return "<br>".join(response_parts)

# --- 4. THE ROUTER ---
def get_ai_response(query: str):
    # This function is unchanged
    _load_static_data()
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

print("Agent module loaded. Models and data will be loaded on first request.")
