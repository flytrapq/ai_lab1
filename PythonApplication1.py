import tkinter as tk  # Importing tkinter library as tk for GUI
from tkinter import ttk  # Importing ttk module from tkinter for themed widgets
from tkinter import scrolledtext  # Importing scrolledtext module from tkinter for scrolling text widget
from tkinter import messagebox  # Importing messagebox module from tkinter for displaying message boxes
import re  # Importing re module for regular expressions
import csv  # Importing csv module for reading and writing CSV files
import nltk  # Importing Natural Language Toolkit (nltk)
from nltk.tokenize import word_tokenize  # Importing word_tokenize function for tokenization
from nltk.corpus import stopwords  # Importing stopwords module from nltk.corpus
from nltk.stem import PorterStemmer  # Importing PorterStemmer from nltk.stem for stemming

nltk.download('punkt')  # Downloading necessary nltk resources
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters and punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Get English stopwords
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()  # Initialize Porter Stemmer
    tokens = [stemmer.stem(token) for token in tokens]  # Perform stemming
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

# Function triggered by the search button
def search():
    keyword = keyword_entry.get()
    num_display = int(display_entry.get())
    keyword_tokens = preprocess_text(keyword)
    relevant_books = search_books_by_keyword(keyword, sample_dataset, category_patterns)
    display_books(relevant_books, num_display, keyword_tokens)

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
num_samples = 500  # Number of samples to take from the dataset

# Read the entire dataset
full_dataset = read_dataset(file_path)

# Take a subset of the dataset for testing
sample_dataset = full_dataset[:num_samples]

# GUI setup
root = tk.Tk()  # Creating tkinter window
root.title("Book Search")  # Setting title of the window
root.geometry("600x400")  # Setting dimensions of the window

keyword_label = ttk.Label(root, text="Enter keyword:")  # Label for keyword entry
keyword_label.pack()
keyword_entry = ttk.Entry(root, width=50)  # Text entry for keyword
keyword_entry.pack()

display_label = ttk.Label(root, text="Enter number of books to display:")  # Label for number of books entry
display_label.pack()
display_entry = ttk.Entry(root, width=10)  # Text entry for number of books
display_entry.pack()

search_button = ttk.Button(root, text="Search", command=search)  # Button to trigger search function
search_button.pack()

result_text = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)  # Scrolled text widget for displaying results
result_text.pack()

root.mainloop()  # Running the tkinter event loop
