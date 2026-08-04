from string import punctuation

from cli.lib.search_utils import load_movies, Movie


def preprocess_text(text: str):
    text = text.lower()
    punctuation_translation_table = str.maketrans("","",punctuation)
    text = text.translate(punctuation_translation_table)
    return text

def search_command(query: str, limit: int = 5) -> list[Movie]:
    movies = load_movies()
    results = []
    processed_query = preprocess_text(query)
    for movie in movies:
        if processed_query in preprocess_text(movie["title"]):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
