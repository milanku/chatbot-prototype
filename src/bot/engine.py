from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from bot.logging import log_event
from bot.models.memory import SessionState, TxQAQueryResult
from bot.models.repository import TransactionsRepository, TxFilter
from bot.models.responses import BotResponse
from bot.models.routing import Recipe, RouterDecision
from bot.recipes.doc_qa import answer
from bot.recipes.tx_qa import parse
from bot.recipes.tx_qa.compute import compute_total_spent
from bot.routing.router import route


@dataclass(frozen=True)
class EngineConfig:
    # TODO add: docs path, model names, retrieval parameters, etc.
    app_name: str = "chatbot-prototype"


@dataclass(frozen=True)
class EngineDeps:
    tx_repository: TransactionsRepository
    tx_history: SessionState


class ChatbotEngine:
    def __init__(self, config: EngineConfig, deps: EngineDeps) -> None:
        self._config = config
        self._deps = deps

    def answer(self, message: str, *, session_id: str = "default") -> BotResponse:
        trace_id = uuid4().hex

        # Trace: engine start
        log_event(
            trace_id=trace_id,
            event="engine.start",
            payload={"session_id": session_id, "message": message, "app": self._config.app_name},
        )

        router_decision: RouterDecision = route(message)

        log_event(
            trace_id=trace_id,
            event="router.decision",
            payload={
                "recipe": router_decision.recipe.value,
                "confidence": router_decision.confidence,
            },
        )

        match router_decision.recipe:
            case Recipe.TX_SUMMARY:
                parsed_query = parse.parse_query(message)
                if parsed_query is None:
                    answer_text = "Sorry, I couldn't understand your query. Please make sure to include a label (food, pets, other) and a date range (e.g., 2026-01-01 - 2026-01-31).\n"
                else:
                    tx_filter = TxFilter(
                        label=parsed_query.label,
                        start=parsed_query.start,
                        end=parsed_query.end,
                        direction="spend",  # For simplicity, we only consider spending transactions in this example
                    )
                    txs = self._deps.tx_repository.list_transactions(tx_filter)
                    total_spent = compute_total_spent(txs)

                    # Save query to history
                    self._deps.tx_history.txs_results.append(
                        TxQAQueryResult(
                            query=parsed_query,
                            txs=txs,
                            total=total_spent,
                        )
                    )

                    answer_text = f"You spent a total of ${total_spent:.2f} on {parsed_query.label} from {parsed_query.start} to {parsed_query.end}.\n"

            case Recipe.TX_EXPLAIN:
                if not self._deps.tx_history.txs_results or self._deps.tx_history.txs_results[-1].query is None:
                    answer_text = "Sorry, I don't have any transaction summary to explain. Please ask a question about your spending first (e.g., 'How much did I spend on food last month?').\n"
                else: 
                    answer_text = f"Here are the transactions that contributed to this (${self._deps.tx_history.txs_results[-1].total:.2f}) sum:\n"
                    for tx in self._deps.tx_history.txs_results[-1].txs:
                        answer_text += (
                            f"- {tx.date}: ${tx.amount:.2f} to {tx.other_account} ({tx.description})\n"
                        )

            case Recipe.DOCS_ANSWER:
                answer_text = "DOCS_ANSWER is not implemented yet.\n"
            case Recipe.OUT_OF_SCOPE:
                answer_text = "Sorry, I can't help with that.\n"
            case _:
                answer_text = "Unknown recipe.\n"

        resp = BotResponse(answer=answer_text, references=[], trace_id=trace_id)

        # Trace: engine finish
        log_event(
            trace_id=trace_id,
            event="engine.finish",
            payload={"references": resp.references},
        )

        return resp
