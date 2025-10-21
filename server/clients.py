from typing import List, Union
import dspy
import janus

class EmittableOllamaLocal(dspy.OllamaLocal):
    def __init__(self, queue: janus._SyncQueueProxy, **kwargs):
        super().__init__(**kwargs)
        self.queue = queue
        self.lm_client = dspy.OllamaLocal(model="llama3:instruct", max_tokens=20)

    def basic_request(self, prompt, **kwargs):
        print("Streaming LM called with request", prompt)
        self.queue.put({"source": "server", "dest": "llama3", "query": prompt})
        # self.loop.call_soon_threadsafe(self.queue.put, str({"source": "server", "dest": "llama3", "query": prompt}))

        try:
            response = self.lm_client(prompt, **kwargs)
        except Exception as e:
            response = [str(e)]
        
        # self.loop.call_soon_threadsafe(self.queue.put, str({"source": "llama3", "dest": "server", "message": response[0]}))
        
        self.queue.put({"source": "llama3", "dest": "server", "message": response[0]})
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return self.request(prompt, **kwargs)

class EmittableAWSAnthropic(dspy.AWSAnthropic):
    def __init__(self, queue: janus._SyncQueueProxy, **kwargs):
        super().__init__(**kwargs)
        self.queue = queue
        self.lm_client = dspy.AWSAnthropic(**kwargs)

    def basic_request(self, prompt, **kwargs):
        print("Streaming LM called with request", prompt)
        # loop = asyncio.get_event_loop()
        # loop.call_soon_threadsafe(self.queue.put_nowait, str({"source": "server", "dest": "llama3", "query": prompt}))
        self.queue.put_nowait({"source": "server", "dest": "haiku", "query": prompt})

        try:
            response = self.lm_client(prompt, **kwargs)
        except Exception as e:
            response = [str(e)]
        # loop.call_soon_threadsafe(self.queue.put_nowait, str({"source": "llama3", "dest": "server", "message": response[0]}))
        self.queue.put_nowait({"source": "haiku", "dest": "server", "message": response[0]})
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return self.request(prompt, **kwargs)

class EmittableAWSMistral(dspy.AWSMistral):
    def __init__(self, queue: janus._SyncQueueProxy, **kwargs):
        super().__init__(**kwargs)
        self.queue = queue
        self.lm_client = dspy.AWSMistral(**kwargs)
    
    def basic_request(self, prompt, **kwargs):
        print("Streaming LM called with request", prompt)
        self.queue.put_nowait({"source": "server", "dest": "mistral", "query": prompt})
        try:
            response = self.lm_client(prompt, **kwargs)
        except Exception as e:
            response = [str(e)]
        self.queue.put_nowait({"source": "mistral", "dest": "server", "message": response[0]})
        return response[0]

class EmittableColBERTv2(dspy.ColBERTv2):
  def __init__(self, queue: janus._SyncQueueProxy, **kwargs):
    super().__init__(**kwargs)
    self.queue = queue
    self.rm_client = dspy.ColBERTv2(**kwargs)
  
  def __call__(self, query: Union[str, List[str]], k:int) -> Union[List[str], List[List[str]]]:
    print("RM called with query", query)
    self.queue.put({"source": "server", "dest": "colbert", "query": query})
    try:
      response = self.rm_client(query, k)
    except Exception as e:
      response = [str(e)]
    self.queue.put({"source": "colbert", "dest": "server", "message": response})
    return response
  