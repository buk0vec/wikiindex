#!/bin/bash

pip install -r server-requirements.txt

DUMP_URL="https://dumps.wikimedia.org/other/enterprise_html/runs/20240401/enwiki-NS0-20240401-ENTERPRISE-HTML.json.tar.gz"
CHECKPOINT_LOCATION="/datasets/nbukovec/wikipedia/colbert/checkpoint/colbertv2.0"

mkdir build

if [[ -z "$DUMP_LOCATION" && -z "$COLLECTION_LOCATION" ]]; then
  DUMP_LOCATION="dump.tar.gz";
  rm $DUMP_LOCATION
  echo "Dowloading dump from $DUMP_URL into $DUMP_LOCATION";
  DUMP_LOCATION="dump.tar.gz";
  wget -c -O $DUMP_LOCATION $DUMP_URL;
fi

mkdir build/colbert; 

if [[ -z "$COLLECTION_LOCATION" ]]; then
  COLLECTION_LOCATION="build/colbert/collection.tsv";
  python3 scripts/process-dump.py $DUMP_LOCATION $COLLECTION_LOCATION; 

else
  cp $COLLECTION_LOCATION build/colbert/$(basename $COLLECTION_LOCATION);
  COLLECTION_LOCATION=build/colbert/$(basename $COLLECTION_LOCATION);
fi

mkdir build/colbert/checkpoint
cp -a "$CHECKPOINT_LOCATION/." build/colbert/checkpoint
CHECKPOINT_LOCATION=build/colbert/checkpoint

if [[ -z "$COLBERT_ROOT" ]]; then
  COLBERT_ROOT="build/colbert";
  INDEX_NAME=${INDEX_NAME:-"index-$(date +%s).nbits=2"};
  python3 scripts/index.py $COLLECTION_LOCATION $COLBERT_ROOT $INDEX_NAME;
fi

SED_REPLACEMENT=$(printf "%s" "$(dirname "$COLBERT_ROOT")" | sed 's#/#\\/#g')
for file in "$COLBERT_ROOT"/index/indexes/*/*.json ; 
do
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' -e "s#$SED_REPLACEMENT##g" $file
  else
    sed -i -e "s#$SED_REPLACEMENT##g" $file
  fi
done;

if [[ -z "$NO_DOCKER" ]]; then 
  docker build --build-arg INDEX_NAME=$INDEX_NAME .;
fi