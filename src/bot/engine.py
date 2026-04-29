from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from bot.llm import llm_client
from bot.logging import log_event
from bot.models.memory import SessionState, TxQAQueryResult
from bot.models.repository import DocRepository, TransactionsRepository, TxFilter
from bot.models.responses import BotResponse
from bot.models.routing import Route, RouterDecision
from bot.routes.doc_qa.synthesize import synthesize_doc_answer
from bot.routes.doc_qa.verify import filter_relevant_hits
from bot.routes.tx_qa import parse
from bot.routes.tx_qa.compute import compute_total_spent
from bot.routes.tx_qa.explain import (
    TXExplainParseIntermediateResult,
    parse_explain_query,
)
from bot.routes.tx_qa.synthesize import synthesize_tx_summary
from bot.routing.router import route


@dataclass(frozen=True)
class EngineConfig:
    # TODO add: docs path, model names, retrieval parameters, etc.
    app_name: str = "chatbot-prototype"


@dataclass(frozen=True)
class EngineDeps:
    tx_repository: TransactionsRepository
    doc_repository: DocRepository
    llm_client: llm_client.LLMClient
    

class ChatbotEngine:
    def __init__(self, config: EngineConfig, deps: EngineDeps) -> None:
        self._config = config
        self._deps = deps
        self._docs = deps.doc_repository
        
    def answer(
        self, message: str, *, session_id: str, session_state: SessionState
    ) -> tuple[BotResponse, SessionState]:
        trace_id = uuid4().hex
        
        new_state = session_state  # By default, the state doesn't change. Routes can override this if needed.

        def trace(event: str, **payload: object) -> None:
             log_event(trace_id=trace_id, event=event, payload=payload)

        # Trace: engine start
        trace(
            "engine.start",
            session_id=session_id,
            message=message,
            app=self._config.app_name,
        )

        router_decision: RouterDecision = route(session_id=session_id, llm_client=self._deps.llm_client, message=message)

        trace(
            "router.decision",
            route=router_decision.route.value,
            confidence=router_decision.confidence,
        )

        match router_decision.route:
            case Route.TX_SUMMARY:
                parsed_query = parse.parse_query(session_id=session_id, llm_client=self._deps.llm_client, msg=message)
                                
                if parsed_query is None:
                    trace(
                        "tx_qa.parse_query",
                        ok=False,
                    )
                    answer_text = "Sorry, I couldn't understand your query. Please make sure to include a label (food, pets, other) and a date range (e.g., 2026-01-01 - 2026-01-31).\n"
                else:
                    trace(
                        "tx_qa.parse_query",
                        ok=True,
                        label=parsed_query.label,
                        start=parsed_query.start.isoformat(),
                        end=parsed_query.end.isoformat(),
                    )
                    tx_filter = TxFilter(
                        label=parsed_query.label,
                        start=parsed_query.start,
                        end=parsed_query.end,
                        direction="spend",  # For simplicity, we only consider spending transactions in this example
                    )
                    trace(
                        "tx_qa.query",
                        label=parsed_query.label,
                        start=parsed_query.start.isoformat(),
                        end=parsed_query.end.isoformat(),
                        direction=tx_filter.direction,
                    )
                    
                    txs = self._deps.tx_repository.list_transactions(tx_filter)
                    total_spent = compute_total_spent(txs)
                    
                    trace(
                        "tx_qa.query_result",
                        total_spent=f"{total_spent:.2f}",
                        num_transactions=len(txs),
                    )
                    query_result: TxQAQueryResult = TxQAQueryResult(
                        query=parsed_query,
                        total=Decimal(total_spent), 
                        created_at=datetime.now(
                            timezone.utc
                        ),  # Using current UTC time as a timestamp
                    )
                    # Update state with the new query result
                    new_state = replace(
                        session_state,
                        txs_results=session_state.txs_results + (query_result,),
                    )
                    answer_text = synthesize_tx_summary(parsed_query, total_spent)

            case Route.TX_EXPLAIN:
                txs_results = session_state.txs_results
                
                reference_result:TXExplainParseIntermediateResult = parse_explain_query(session_id=session_id, llm_client=self._deps.llm_client, msg=message)
                
                trace(
                    "tx_explain.parse_query",
                    txs_results=len(txs_results),
                    reference_offset=reference_result.reference_offset,
                    reference_count=reference_result.reference_count,
                    confidence=reference_result.confidence,
                    reason=reference_result.reason,
                )
                
                related_results = []
                if reference_result.reference_offset is not None and reference_result.reference_count is not None:
                    offset = reference_result.reference_offset
                    count = reference_result.reference_count
                    related_results = txs_results[max(0, len(txs_results) - offset - count):len(txs_results) - offset]
                    trace(
                        "tx_explain.find_reference",
                        found=bool(related_results),
                        reference_offset=reference_result.reference_offset,
                        reference_count=reference_result.reference_count,
                        confidence=reference_result.confidence,
                        reason=reference_result.reason,
                    )
                else:
                    trace(
                        "tx_explain.find_reference",
                        found=False,
                        confidence=reference_result.confidence,
                        reason=reference_result.reason,
                    )

                if not related_results:
                    answer_text = "Sorry, I don't have any transaction summary to explain. Please ask a question about your spending first (e.g., 'How much did I spend on food last month?').\n"
                else:
                    answer_text = "Here are the transactions that contributed to your selected sum:\n"
                    for tx_result in related_results:
                        txs = self._deps.tx_repository.list_transactions(
                            TxFilter(
                                label=tx_result.query.label,
                                start=tx_result.query.start,
                                end=tx_result.query.end,
                                direction="spend",
                            )
                        )
                        for tx in txs:
                            answer_text += f"- {tx.date}: {tx.amount:.2f} EUR to {tx.other_account} ({tx.description})\n"

            case Route.DOCS_ANSWER:
                top_k_chunks = self._deps.doc_repository.get_top_k_chunks(message, top_k=5)
                trace(
                    "doc_qa.retrieval",
                    query=message,
                    retrieved_chunks=[{"id": hit.id, "score": hit.score, "content": hit.content} for hit in top_k_chunks]
                )
                filtered_hits = filter_relevant_hits(hits=top_k_chunks, absolute_relevance_threshold=0.5, relative_relevance_threshold=0.85)
                trace(
                    "doc_qa.relevance_filter",
                    query=message,
                    retrieved_chunks=len(top_k_chunks),
                    relevant_chunks=len(filtered_hits),
                )
                answer_text = synthesize_doc_answer(llm_client=self._deps.llm_client, question=message, hits=filtered_hits)
            case Route.OUT_OF_SCOPE:
                answer_text = "Sorry, I can't help with that.\n"
            case _:
                answer_text = "Unknown route.\n"

        bot_response = BotResponse(answer=answer_text, references=[], trace_id=trace_id)

        # Trace: engine finish
        log_event(
            trace_id=trace_id,
            event="engine.finish",
            payload={"references": bot_response.references},
        )

        return bot_response, new_state        