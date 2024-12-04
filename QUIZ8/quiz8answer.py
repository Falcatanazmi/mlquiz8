import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def luhn_summarization(text, summary_length=3):
    # Preprocessing
    stop_words = set(stopwords.words("english"))
    sentences = sent_tokenize(text)
    word_counts = Counter(word.lower() for word in word_tokenize(text) if word.isalnum())
    
    # Compute significant words
    significant_words = {word for word, count in word_counts.items() if count > 1 and word not in stop_words}
    
    # Score sentences based on clusters of significant words
    sentence_scores = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        sig_word_indices = [i for i, word in enumerate(words) if word in significant_words]
        if not sig_word_indices:
            continue
        
        # Find clusters of significant words
        clusters = []
        cluster = [sig_word_indices[0]]
        for index in sig_word_indices[1:]:
            if index - cluster[-1] <= 4:  # Allowable distance in a cluster
                cluster.append(index)
            else:
                clusters.append(cluster)
                cluster = [index]
        clusters.append(cluster)
        
        # Compute score for each cluster
        max_cluster_score = 0
        for cluster in clusters:
            cluster_size = cluster[-1] - cluster[0] + 1
            significant_word_count = len(cluster)
            cluster_score = significant_word_count**2 / cluster_size
            max_cluster_score = max(max_cluster_score, cluster_score)
        
        sentence_scores.append((max_cluster_score, sentence))
    
    # Sort sentences by score and select top ones
    sentence_scores.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sentence for _, sentence in sentence_scores[:summary_length]]
    return ' '.join(top_sentences)

if __name__ == "__main__":
    with open("blog_post.txt.txt", "r") as file:
        blog_post_text = file.read()

summary = luhn_summarization(blog_post_text)
print(summary),
