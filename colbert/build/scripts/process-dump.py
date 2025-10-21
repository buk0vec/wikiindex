"""
process-dump.py
By Nick Bukovec
Processing enterprise wikipedia dumps into chunked CSVs for indexing
"""

import json
import os
import sys
import tarfile
import csv
from pathlib import Path
import time
from tqdm import tqdm
from transformers import BertTokenizerFast
from multiprocessing import Queue, Value, Process
from typing import List

from mwparserfromhtml.dump.dump import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

if len(sys.argv) < 3:
  print("Missing dataset location and/or output location")
  sys.exit(1)

DATASET_LOCATION = sys.argv[1]
OUTPUT_LOCATION = sys.argv[2]

MAX_INPUT_QUEUE_SIZE = 1024
MAX_OUTPUT_QUEUE_SIZE = 1024

BREAKPOINT = 6945585 # From manual count, isn't necessary but I like progress bars :)

# BREAKPOINT = 3000

NUM_WORKERS = 20

SENTINEL = "SENTINEL VALUE -- THERE IS NO WAY THIS ACTUALLY IS INCLUDED IN A WIKIPEDIA ARTICLE"

class HTMLDump:
    """
    Class file to create instances of Wikimedia Enterprise HTML Dumps
    Stolen from mwparserfromhtml repo, with some modifications for my usecase (return entires w/o parsing):
    https://gitlab.wikimedia.org/repos/research/html-dumps/-/blob/main/src/mwparserfromhtml/dump/dump.py
    """

    def __init__(self, filepath: str = None, fileobj=None) -> None:
        """
        Constructor for HTMLDump class
        """
        assert filepath is not None or fileobj is not None

        if fileobj is not None:
            self.size = -1.0
            if filepath is None:
                # fileobj may not have useful path info, so we'll only
                # use it if no filename is explicitly provided
                self.database = str(Path(fileobj.name).name).split("-")[0]

            self.tarfile_open_args = {
                "name": None,
                "fileobj": fileobj,
                "mode": f"{fileobj.mode.strip('b+')}:gz",
            }
        elif filepath is not None:
            self.size = os.path.getsize(filepath) / (1024 * 1024 * 1024)
            self.database = str(Path(filepath).name).split("-")[0]

            self.tarfile_open_args = {
                "name": filepath,
                "fileobj": None,
                "mode": "r:gz",
            }

    def __str__(self) -> str:
        """
        String representation of the HTMLDump class
        """
        return f" HTMLDump (database = {self.database}, size = {self.size} GB"

    def __repr__(self) -> str:
        """
        String representation of the HTMLDump class
        """
        return str(self)

    def __iter__(self):
        """
        Iterator of the Article class
        """
        return self.read_dump()

    def read_dump(self):
        """
        Reads a dump file and returns an iterator of the rows.
        Returns:
            Iterator[List[Any]]: iterator of the rows
        """

        tar_file_ = tarfile.open(**self.tarfile_open_args)
        count = 0
        while True:
            html_fn = tar_file_.next()
            if html_fn is None:
                tar_file_.close()
                return

            else:
                with tar_file_.extractfile(html_fn) as file_input:
                    for line in file_input:
                        article = json.loads(line)
                        count += 1
                        try:
                            # yield Document(article)
                            yield article
                        except Exception:
                            print(f"Article parsing failed for: {article}")
                            continue

def get_content(d: Document):
    """
    Given a Document, return back all of the content that needs to be included in the document as a string.
    """
    return d.html.wikistew.get_first_paragraph()

def len_fn(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunkify_content(content: str, tokenizer, prefix: str=""):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=400 - len_fn(prefix, tokenizer),
        chunk_overlap=20,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    chunks = [c.page_content for c in text_splitter.create_documents([content])]
    chunks = [prefix + c.replace('\n', ' ') for c in chunks]
    return chunks
  
def reader_process(input_queue: Queue, output_queue: Queue, process_counter):
    # print("Reader process started!")
    from mwparserfromhtml.dump.dump import Document
    from transformers import BertTokenizerFast

    # print("Loading tokenizer... ")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # print("Done")
    while True:
        line = input_queue.get(timeout=3000)
        if line == SENTINEL:
            # print(input_queue.qsize(), output_queue.qsize(), process_counter.value)
            return
        try:
            d = Document(line)
        except:
            print("Parse filed on line", line)
        content = get_content(d)
        if content != None:
            chunks = chunkify_content(content, tokenizer, prefix=d.get_title() + " | ")
            for chunk in chunks:
                output_queue.put(chunk)
        with process_counter.get_lock():
            process_counter.value -= 1
    
def writer_process(output_queue: Queue):
    print("Writer process start")
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    passage_id = 0
    sentinel_seen = False
    with open(OUTPUT_LOCATION, 'w') as f:
        print("Starting to write to", OUTPUT_LOCATION)
        fw = csv.writer(f, delimiter='\t', lineterminator='\n')
        # Replaced the while True, for some reason my sentinel value sent from the main process is getting
        # higher priority in the queue than the ones sent by the reader process
        pbar = tqdm(total=BREAKPOINT, position=1)
        while (not sentinel_seen) or output_queue.qsize() > 0:
            entry = output_queue.get()
            if entry == SENTINEL: 
                sentinel_seen = True
                continue 
            fw.writerow([passage_id, entry])
            passage_id += 1
            # print(output_queue.qsize(), flush=True)
            pbar.set_description("Write progress (estimated)")
            pbar.update()
    print(f"Writer process complete, wrote {passage_id} values")

def process_parallel():
    input_queue = Queue()
    output_queue = Queue()
    process_counter = Value('i', 0)
    dump = HTMLDump(filepath=DATASET_LOCATION)
    
    processes: List[Process] = []
    
    write_proc = Process(target=writer_process, args=(output_queue,))
    write_proc.start()
    
    for j in range(NUM_WORKERS):
        # print("Spawning process " + str(j))
        p = Process(target=reader_process,args=(input_queue, output_queue, process_counter))
        processes.append(p)
        p.start()
    
    i = 0
    for line in tqdm(dump, total=BREAKPOINT, position=0):
        if line == None:
            continue
        input_queue.put(line)
        with process_counter.get_lock():
            process_counter.value += 1
        if i > BREAKPOINT:
            break
        i += 1

    for _ in range(NUM_WORKERS):
        input_queue.put(SENTINEL)

    print("Waiting for workers...")
    while process_counter.value > 0:
        pass
    
    print("Done waiting for workers")

    output_queue.put(SENTINEL)
    
    write_proc.join()
    write_proc.close()
    
    for p in processes:
        if p.is_alive():
            p.join()
            p.close()
        else:
            p.close()    
    
    print("Processes closed")
    
if __name__ == "__main__":
    print(f"Starting parse of {BREAKPOINT}...")
    process_parallel()
    print("Done")