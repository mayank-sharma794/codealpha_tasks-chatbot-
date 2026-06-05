import tkinter as tk
from tkinter import scrolledtext, messagebox
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- NLTK DOWNLOAD ---------------- #

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ---------------- FAQ DATA ---------------- #

faq_data = {
    "What is Python?":
        "Python is a powerful programming language.",

    "Who developed Python?":
        "Python was developed by Guido van Rossum.",

    "What is AI?":
        "AI stands for Artificial Intelligence.",

    "What is NLP?":
        "NLP means Natural Language Processing.",

    "What is Machine Learning?":
        "Machine Learning allows computers to learn from data."
}

# ---------------- NLP PREPROCESSING ---------------- #

stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# ---------------- PREPARE QUESTIONS ---------------- #

questions = list(faq_data.keys())

processed_questions = [
    preprocess_text(q)
    for q in questions
]

# ---------------- TF-IDF MODEL ---------------- #

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)

# ---------------- CHATBOT RESPONSE ---------------- #

def chatbot_response(user_question):

    processed_input = preprocess_text(user_question)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarity.argmax()

    best_score = similarity[0][best_match_index]

    if best_score > 0.2:
        matched_question = questions[best_match_index]
        return faq_data[matched_question]
    else:
        return "Sorry, I don't understand your question."

# ---------------- SEND MESSAGE ---------------- #

def send_message():

    user_message = entry_box.get()

    if user_message.strip() == "":
        messagebox.showwarning(
            "Warning",
            "Please enter a question"
        )
        return

    # Display User Message
    chat_area.insert(
        tk.END,
        "You: " + user_message + "\n"
    )

    # Bot Response
    response = chatbot_response(user_message)

    # Display Bot Response
    chat_area.insert(
        tk.END,
        "Bot: " + response + "\n\n"
    )

    # Auto Scroll
    chat_area.yview(tk.END)

    # Clear Entry
    entry_box.delete(0, tk.END)

# ---------------- ADD FAQ FUNCTION ---------------- #

def add_faq():

    question = new_question.get()
    answer = new_answer.get()

    if question.strip() == "" or answer.strip() == "":
        messagebox.showerror(
            "Error",
            "Please fill both fields"
        )
        return

    # Add FAQ
    faq_data[question] = answer

    # Update Model
    update_model()

    messagebox.showinfo(
        "Success",
        "New FAQ Added Successfully"
    )

    new_question.delete(0, tk.END)
    new_answer.delete(0, tk.END)

# ---------------- UPDATE MODEL ---------------- #

def update_model():

    global questions
    global processed_questions
    global faq_vectors

    questions = list(faq_data.keys())

    processed_questions = [
        preprocess_text(q)
        for q in questions
    ]

    faq_vectors = vectorizer.fit_transform(
        processed_questions
    )

# ---------------- GUI ---------------- #

root = tk.Tk()

root.title("NLP FAQ Chatbot")
root.geometry("750x650")
root.config(bg="#F7DC6F")

# ---------------- TITLE ---------------- #

title = tk.Label(
    root,
    text="NLP FAQ Chatbot",
    font=("Helvetica", 22, "bold"),
    bg="#F7DC6F"
)

title.pack(pady=10)

# ---------------- CHAT AREA ---------------- #

chat_area = scrolledtext.ScrolledText(
    root,
    width=80,
    height=20,
    font=("Arial", 11),
    wrap=tk.WORD
)

chat_area.pack(padx=10, pady=10)

# ---------------- ENTRY BOX ---------------- #

entry_box = tk.Entry(
    root,
    width=50,
    font=("Arial", 14)
)

entry_box.pack(pady=10)

# ---------------- SEND BUTTON ---------------- #

send_button = tk.Button(
    root,
    text="Send",
    font=("Arial", 12, "bold"),
    bg="#248aa2",
    fg="white",
    width=15,
    command=send_message
)

send_button.pack(pady=5)

# ---------------- ADD FAQ SECTION ---------------- #

add_label = tk.Label(
    root,
    text="Add New FAQ",
    font=("Arial", 16, "bold"),
    bg="#F7DC6F"
)

add_label.pack(pady=10)

# ---------------- NEW QUESTION ---------------- #

new_question = tk.Entry(
    root,
    width=60,
    font=("Arial", 12)
)

new_question.pack(pady=5)
new_question.insert(0, "Enter New Question")

# ---------------- NEW ANSWER ---------------- #

new_answer = tk.Entry(
    root,
    width=60,
    font=("Arial", 12)
)

new_answer.pack(pady=5)
new_answer.insert(0, "Enter Answer")

# ---------------- ADD BUTTON ---------------- #

add_button = tk.Button(
    root,
    text="Add FAQ",
    font=("Arial", 12, "bold"),
    bg="green",
    fg="white",
    width=15,
    command=add_faq
)

add_button.pack(pady=10)

# ---------------- RUN APPLICATION ---------------- #

root.mainloop()