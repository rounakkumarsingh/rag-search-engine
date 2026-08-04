from cli.lib.search_utils import load_movies, Movie

def search_command(query: str, limit: int = 5) -> list[Movie]:
    movies = load_movies()
    results = []
    for movie in movies:
        if query in movie["title"]:
            results.append(movie)
            if len(results) >= limit:
                break
    return results
