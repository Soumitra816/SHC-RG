import json
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from sentence_transformers import SentenceTransformer, util as sentemb

# Load the test dataset
with open('test_set.json', 'r') as file:
    test_set = json.load(file)

# Load the predictions
with open('predictions.json', 'r') as file:
    predictions = json.load(file)

# Define functions to compute various metrics
def compute_relevance(rationale, casual_mentions, serious_intents):
    combined_spans = casual_mentions + serious_intents
    spans_text = " ".join(combined_spans).lower()
    rationale_lower = rationale.lower()
    relevance = all(span.lower() in rationale_lower for span in combined_spans)
    return relevance

def compute_coherence(rationale, casual_mentions, serious_intents):
    combined_spans = casual_mentions + serious_intents
    spans_text = " ".join(combined_spans)
    vectorizer = TfidfVectorizer().fit_transform([rationale, spans_text])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return similarity

def compute_readability(rationale):
    readability_score = textstat.flesch_kincaid_grade(rationale)
    return readability_score

def compute_f1_for_spans(pred_spans, true_spans):
    true_positive = len(set(pred_spans).intersection(set(true_spans)))
    false_positive = len(set(pred_spans) - set(true_spans))
    false_negative = len(set(true_spans) - set(pred_spans))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def compute_SENTSIM(rationale, casual_mentions, serious_intents):
    combined_spans = casual_mentions + serious_intents
    spans_text = " ".join(combined_spans)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rationale_embedding = model.encode(rationale)
    spans_embedding = model.encode(spans_text)
    similarity = sentemb.cosine_similarity(rationale_embedding, spans_embedding)
    return similarity[0][0]

# Initialize lists to store metrics
classification_labels = []
classification_preds = []
casual_mentions_labels = []
casual_mentions_preds = []
serious_intents_labels = []
serious_intents_preds = []
relevance_scores = []
coherence_scores = []
readability_scores = []
sent_sim_scores = []

# Iterate through the test set and predictions to compute metrics
for sample, prediction in zip(test_set, predictions):
    true_label = 1 if sample['label'] == 'self-harm' else 0
    pred_label = 1 if prediction['classification'] == 'self-harm' else 0
    
    classification_labels.append(true_label)
    classification_preds.append(pred_label)
    
    true_casual_mentions = sample.get('casual_mention_spans', [])
    pred_casual_mentions = prediction.get('casual_mention_spans', [])
    casual_mentions_labels.extend(true_casual_mentions)
    casual_mentions_preds.extend(pred_casual_mentions)
    
    true_serious_intents = sample.get('serious_intent_spans', [])
    pred_serious_intents = prediction.get('serious_intent_spans', [])
    serious_intents_labels.extend(true_serious_intents)
    serious_intents_preds.extend(pred_serious_intents)
    
    rationale = prediction.get('rationale', '')
    relevance_scores.append(compute_relevance(rationale, true_casual_mentions, true_serious_intents))
    coherence_scores.append(compute_coherence(rationale, true_casual_mentions, true_serious_intents))
    readability_scores.append(compute_readability(rationale))
    sent_sim_scores.append(compute_SENTSIM(rationale, true_casual_mentions, true_serious_intents))

# Compute F1 scores for classification
classification_f1 = f1_score(classification_labels, classification_preds)
classification_precision = precision_score(classification_labels, classification_preds)
classification_recall = recall_score(classification_labels, classification_preds)

# Compute F1 scores for spans
casual_mentions_f1 = compute_f1_for_spans(casual_mentions_preds, casual_mentions_labels)
serious_intents_f1 = compute_f1_for_spans(serious_intents_preds, serious_intents_labels)

# Compute average relevance, coherence, readability, and sentence similarity scores
avg_relevance = sum(relevance_scores) / len(relevance_scores)
avg_coherence = sum(coherence_scores) / len(coherence_scores)
avg_readability = sum(readability_scores) / len(readability_scores)
avg_sent_sim = sum(sent_sim_scores) / len(sent_sim_scores)

# Print the metrics
print(f"Classification F1 Score: {classification_f1:.4f}")
print(f"Classification Precision: {classification_precision:.4f}")
print(f"Classification Recall: {classification_recall:.4f}")
print(f"Casual Mentions F1 Score: {casual_mentions_f1:.4f}")
print(f"Serious Intents F1 Score: {serious_intents_f1:.4f}")
print(f"Average Relevance Score: {avg_relevance:.4f}")
print(f"Average Coherence Score: {avg_coherence:.4f}")
print(f"Average readability Score: {avg_readability:.4f}")
print(f"Average Sentence Similarity Score: {avg_sent_sim:.4f}")

