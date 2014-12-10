class Document:
    """
    Document model for the General Conference data set
    """
    def __init__(self, raw):
        lines = raw.split("\n")
        # I'm not sure how reliable this actually is
        self.speaker = lines[0][9:].split("\n")[0]
        self.gender = lines[2][8:].split("\n")[0]
        self.title = lines[3][7:].split("\n")[0]
        self.topics = lines[4][7:].split("/")
        self.link = lines[9][6:].split("\n")[0]
        self.content = raw[raw.find("\n\n")+2:]
        self.content_lower = self.content.lower()

    def count(self, word):
        # Ew, count on string?
        return self.content_lower.count(str(word))

