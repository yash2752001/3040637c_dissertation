import os
import pyterrier as pt

def main():
    if not pt.started():
        pt.java.init()

    dataset = pt.datasets.get_dataset("vaswani")
    corpus = dataset.get_corpus()

    def corpus_gen():
        for i, doc in enumerate(corpus):
            yield {"docno": f"doc_{i}", "text": doc}

    index_path = os.path.abspath(os.path.join(os.getcwd(), "var", "vaswani_index"))
    os.makedirs(index_path, exist_ok=True)

    indexer = pt.IterDictIndexer(index_path)
    print("Indexing documents...")
    indexref = indexer.index(corpus_gen())
    print("Indexing complete!")

if __name__ == "__main__":
    main()
