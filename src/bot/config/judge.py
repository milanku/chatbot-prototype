from pydantic import BaseModel


class ChunkRequirementJudgeConfig(BaseModel):
    retries: int = 3
    
class ChunkRelevanceJudgeConfig(BaseModel):
    retries: int = 3
    
JudgeConfig = ChunkRequirementJudgeConfig | ChunkRelevanceJudgeConfig