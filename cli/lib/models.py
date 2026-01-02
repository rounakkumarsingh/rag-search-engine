class Movie:
    def __init__(self, id: int, title: str, description: str):
        self.id = id
        self.title = title
        self.description = description

    def doc_id(self) -> int:
        return self.id

    def __str__(self) -> str:
        return f"{self.title} {self.description}"


from typing import TypedDict


class MovieJSON(TypedDict):
    id: int
    title: str
    description: str


class Root(TypedDict):
    movies: list[MovieJSON]
