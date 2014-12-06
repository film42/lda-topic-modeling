import shutil

"""
This Data Cleaner is made to clean the general conference NLP data
"""


class DataCleaner:
    def __init__(self, index):
        self.index_path = index
        self.index = self.get_index_list()

    def is_topic(self, line):
        if line[:-1] == "TOPIC: ":
            return False
        if line[:-1] == "TOPIC: None":
            return False
        else:
            return True

    def get_index_list(self):
        with open(self.index_path, 'r') as f:
            return f.read().split()

    def select_with_valid_topic(self, prefix):
        for document_path in self.index:
            with open("%s/%s" % (prefix, document_path), 'r') as f:
                line = ""
                while line[:6] != "TOPIC:":
                    line = f.readline()

                # Now we have the topic line
                if self.is_topic(line):
                    src = "%s/%s" % (prefix, document_path)
                    name = document_path.split("\n")[0].split("/")[-1]
                    dest = "%s/%s" % ("../data/scoped", name)
                    shutil.copyfile(src, dest)


if __name__ == "__main__":
    dc = DataCleaner("../data/raw/all.txt")
    dc.select_with_valid_topic("../data/raw/")