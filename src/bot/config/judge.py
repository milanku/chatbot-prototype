from pydantic import BaseModel, Field


class ChunkRequirementJudgeConfig(BaseModel):
    retries: int = Field(default=3, gt=0)


class ChunkRelevanceJudgeConfig(BaseModel):
    retries: int = Field(default=3, gt=0)


JudgeConfig = ChunkRequirementJudgeConfig | ChunkRelevanceJudgeConfig
