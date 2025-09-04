from beir import util

dataset = "trec-covid"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"

out_dir = "datasets"

util.download_and_unzip(url, out_dir)
