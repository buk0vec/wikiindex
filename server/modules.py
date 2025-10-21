import dspy 

class AnswerQuestion(dspy.Signature):
  """Answer questions with short factoid answers."""
  question = dspy.InputField(desc="")
  answer = dspy.OutputField(desc="A short, simple sentence.")

class AnswerQuestionFromFacts(dspy.Signature):
    """Answer questions with short factoid answers"""
    context = dspy.InputField(desc="May contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A short, simple sentence.")

class BasicQA(dspy.Module):
  def __init__(self):
      super().__init__()

      self.answer_question = dspy.Predict(AnswerQuestion)

  def forward(self, question):
      prediction = self.answer_question(question=question)
      return dspy.Prediction(answer=prediction.answer)
    
class ChainOfThoughtQA(dspy.Module):
  def __init__(self):
      super().__init__()

      self.answer_question = dspy.ChainOfThought(AnswerQuestion)

  def forward(self, question):
      prediction = self.answer_question(question=question)
      return dspy.Prediction(answer=prediction.answer)

class BasicRAG(dspy.Module):
    def __init__(self, num_passages=10):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(AnswerQuestionFromFacts)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(answer=prediction.answer)
      
class GenerateAnswer(dspy.Signature):
    """
    You are an advanced multi-hop information retrieval system powered by generative AI.
    Answer questions with short factoid answers. You will be given facts you can use to support your answer, although you may not need them based on your vast training data.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
class SummarizeContext(dspy.Signature):
    """
    You are an advanced multi-hop information retrieval system.
    Summarize the context to only contain relevant information to the question. If a fact may be useful later in determining an answer to the question, include it. 
    """
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    summary = dspy.OutputField(desc="facts that may be used in later steps to answer the query")

class GenerateSearchQuery(dspy.Signature):
    """
    You are an advanced multi-hop information retrieval system.
    Write a single simple search query to find more information to help answer a question that requires multiple retrieval hops. This query will be embedded into a vector which will be compared with other embedding vectors, so make sure to include important specific keywords for the query. 
    
    """

    context = dspy.InputField(desc="the information that we already have")
    question = dspy.InputField(desc="the complex question we are trying to answer")
    query = dspy.OutputField() 

class MultiHopSearch(dspy.Module):
    def __init__(self, passages_per_hop=5, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.summarize_context = [dspy.ChainOfThought(SummarizeContext) for _ in range(max_hops)]
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context.append(self.summarize_context[hop](context=passages, question=question).summary)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    