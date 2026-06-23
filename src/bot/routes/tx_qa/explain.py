import json
from pathlib import Path

from bot.llm.llm_client import LLMClient
from bot.logging import log_event
from bot.models.tx_qa.query import TXExplainParseIntermediateResult
from bot.routes.tx_qa.explain_prompt_builder import (
    TXExplainParserPromptInput,
    build_tx_explain_parser_system_prompt,
    build_tx_explain_parser_user_prompt,
    load_tx_explain_parser_instructions,
)


def _deserialize_explain_parse_intermediate_result(raw: str) -> TXExplainParseIntermediateResult:
    data = json.loads(raw)
    return TXExplainParseIntermediateResult(
        reference_offset=data.get("reference_offset"),
        reference_count=data.get("reference_count"),
        confidence=data.get("confidence"),
        reason=data.get("reason"),
    )
    
def parse_explain_query(session_id: str, llm_client: LLMClient, msg: str) -> TXExplainParseIntermediateResult | None:
    prompt_template = load_tx_explain_parser_instructions(Path("src/bot/prompts/explain_parse_instructions.txt"))
    system_prompt = build_tx_explain_parser_system_prompt(
        template=prompt_template
    )
    user_prompt = build_tx_explain_parser_user_prompt(TXExplainParserPromptInput(message=msg))
    
    log_event(
        trace_id=session_id,
        event="explain_parser.input",
        payload={"message": msg}
    )
    
    raw_result = llm_client.generate(
        prompt=user_prompt,
        system_instructions=system_prompt
    )
    
    log_event(
        trace_id=session_id,
        event="explain_parser.output",
        payload={"raw_result": raw_result}
    )
    
    try:
        return _deserialize_explain_parse_intermediate_result(raw_result)
    except Exception as e:
        # Log the error and return None if deserialization fails
        log_event(
            trace_id=session_id,
            event="explain_parser.error",
            payload={"error": str(e)}
        )
        return None