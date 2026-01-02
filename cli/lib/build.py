from cli.lib.inverted_index import InvertedIndex
from cli.lib.models import Movie
from cli.lib.search_utils import load_movies


def build():
    idx = InvertedIndex[Movie]()
    idx.build(load_movies())
    idx.save()
