from pydantic import BaseModel


class Question(BaseModel):
    id: str
    content: str
    reference_answer: str
    
class QuestionCollection(BaseModel):
    id: str
    questions: list[Question]
    
