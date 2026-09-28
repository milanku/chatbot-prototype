from pydantic import BaseModel

from bot.bot_models import BotResponse
from bot.tx_qa.memory.models import SessionState


class EngineResponse(BaseModel):
    response: BotResponse
    new_state: SessionState | None
