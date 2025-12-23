class PageRegistry:
    def __init__(self):
        self._pages = {}

    def register(self, title, page_instance):
        """Registers a page instance with a title."""
        self._pages[title] = page_instance

    def get_page(self, title):
        """Retrieves a page instance by title."""
        return self._pages.get(title)

    def get_titles(self):
        """Returns a list of all registered page titles."""
        return list(self._pages.keys())
