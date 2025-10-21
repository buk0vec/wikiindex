import sys
from pathlib import Path
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
import time

if len(sys.argv) < 4:
  print("Missing collection location and/or checkpoint location and/or colbert root")
  sys.exit(1)

COLLECTION_LOCATION = sys.argv[1]
COLBERT_ROOT = sys.argv[2]
CHECKPOINT_LOCATION = str(Path(COLBERT_ROOT).joinpath("checkpoint"))
INDEX_NAME = f"index-{int(time.time)}.nbits=2" if len(sys.argv) <= 3 else sys.argv[3]

if __name__ == '__main__':
  with Run().context(RunConfig(nranks=1, experiment="index", root=str(COLBERT_ROOT))):

      config = ColBERTConfig(
          nbits=2,
          kmeans_niters=4,
      )
      indexer = Indexer(checkpoint=CHECKPOINT_LOCATION, config=config)
      indexer.index(name=INDEX_NAME, collection=COLLECTION_LOCATION, overwrite=True)
