import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import re
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Function to read dataset from CSV file
def read_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    return dataset

# Function to extract category from text using a specific pattern
def extract_category(text, category_pattern):
    match = re.search(category_pattern, text)
    if match:
        return [item.strip() for item in match.groups() if item]
    return []

# Function to search books by keyword in the dataset
def search_books_by_keyword(keyword, dataset, category_patterns):
    relevant_books = []
    keyword_tokens = preprocess_text(keyword)
    for book in dataset:
        match_score = 0
        for category, pattern in category_patterns.items():
            extracted_info = extract_category(book[0], pattern)
            if extracted_info:
                extracted_tokens = preprocess_text(extracted_info[0])
                match_score += sum(token in extracted_tokens for token in keyword_tokens)
        if match_score > 0:
            relevant_books.append((book, match_score))
    relevant_books.sort(key=lambda x: x[1], reverse=True)
    return relevant_books

# Function to display relevant books
def display_books(relevant_books, num_display, keyword_tokens):
    if relevant_books:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Displaying Top {num_display} Relevant Books:\n\n")
        for i, (book, score) in enumerate(relevant_books[:num_display]):
            result_text.insert(tk.END, f"Book Information {i+1}:\n\n")
            for category, pattern in category_patterns.items():
                extracted_info = extract_category(book[0], pattern)
                if extracted_info:
                    result_text.insert(tk.END, f"{category}: {extracted_info[0]}\n")
            keyword_groups = {}
            for category, pattern in category_patterns.items():
                extracted_info = extract_category(book[0], pattern)
                if extracted_info:
                    extracted_tokens = preprocess_text(extracted_info[0])
                    keyword_groups[category] = [token for token in extracted_tokens if token in keyword_tokens]
            for category, keywords in keyword_groups.items():
                if keywords:
                    result_text.insert(tk.END, f"\n\nKeywords in {category}: {', '.join(keywords)}\n")
            genres = extract_category(book[0], category_patterns["Genres"])
            result_text.insert(tk.END, f"Match Score: {score}\n")
            result_text.insert(tk.END, "-------------------------\n")
    else:
        messagebox.showinfo("No Results", "No relevant books found.")

# Function to display clusters with most repeated tokens without stop words
def display_clusters(dataset):
    vectorizer = CountVectorizer(stop_words='english')  # Exclude English stop words
    X = vectorizer.fit_transform(dataset)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    # Find most repeated tokens in each cluster
    feature_names = vectorizer.get_feature_names_out()
    top_tokens_per_cluster = {}
    for cluster_idx, center in enumerate(cluster_centers):
        top_indices = np.argsort(center)[::-1][:5]  # Get indices of top 5 tokens
        top_tokens_per_cluster[cluster_idx] = [feature_names[i] for i in top_indices]
    
    # Display clusters and their most repeated tokens
    result_text.insert(tk.END, "\n\nTop 5 Clusters with Most Repeated Tokens (Excluding Stop Words):\n\n")
    for cluster_idx, top_tokens in top_tokens_per_cluster.items():
        result_text.insert(tk.END, f"Cluster {cluster_idx + 1}:\n")
        result_text.insert(tk.END, ", ".join(top_tokens) + "\n\n")
        
    # Open a new window to display dataset elements corresponding with the cluster tokens
    new_window = tk.Toplevel(root)
    new_window.title("Dataset Elements for Clusters")
    new_window.geometry("600x400")
    for cluster_idx, top_tokens in top_tokens_per_cluster.items():
        cluster_label = ttk.Label(new_window, text=f"Cluster {cluster_idx + 1} Tokens: {' '.join(top_tokens)}")
        cluster_label.pack()
        for idx, book_info in enumerate(dataset):
            if any(token in preprocess_text(book_info) for token in top_tokens):
                book_info_label = ttk.Label(new_window, text=f"Book {idx + 1}: {book_info}")
                book_info_label.pack()


# Function triggered by the search button
def search():
    keyword = keyword_entry.get()
    num_display = int(display_entry.get())
    keyword_tokens = preprocess_text(keyword)
    relevant_books = search_books_by_keyword(keyword, sample_dataset, category_patterns)
    display_books(relevant_books, num_display, keyword_tokens)

# GUI setup
root = tk.Tk()
root.title("Book Search")
root.geometry("600x400")

keyword_label = ttk.Label(root, text="Enter keyword:")
keyword_label.pack()
keyword_entry = ttk.Entry(root, width=50)
keyword_entry.pack()

display_label = ttk.Label(root, text="Enter number of books to display:")
display_label.pack()
display_entry = ttk.Entry(root, width=10)
display_entry.pack()

search_button = ttk.Button(root, text="Search", command=search)
search_button.pack()

cluster_button = ttk.Button(root, text="Display Clusters", command=lambda: display_clusters([book[0] for book in sample_dataset]))
cluster_button.pack()

result_text = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
result_text.pack()

# Define category patterns for extracting information from book data
category_patterns = {
    "Book Number": r"(\d+)",
    "Book Name": r'/show/\d+\.([^;"\s]+)',
    "Description": r'"((?:[^"]|\\")*)"',
    "Genres": r"\[([^]]+)\]",
    "Average Rating": r"(\d+\.\d+)",
    "URL": r'(https?://\S+(?:[^;"\s]+))'
}

# Example usage:
file_path = 'goodreads_data1.csv'
num_samples = 500

# Read the entire dataset
full_dataset = read_dataset(file_path)

# Take a subset of the dataset for testing
sample_dataset = full_dataset[:num_samples]

root.mainloop()