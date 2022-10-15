from src.loader.class_loader import Dataset


def test_type(dataset: Dataset, source: str):
    dataset.set_source(source)

    all_sent = dataset.clusters[0].get_all_sents()
    print("** Total sent: ", len(all_sent))

    print("** All sentence: ", '\n\t' + '\n\t'.join(dataset.clusters[0].get_all_sents()))
    print("** Golden summary: ", '\n\t' + '\n\t'.join(dataset.clusters[0].get_summary()))