from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from uuid import uuid4

from bot.logging import log_event
from bot.models.memory import SessionState, TxQAQueryResult
from bot.models.repository import TransactionsRepository, TxFilter
from bot.models.responses import BotResponse
from bot.models.routing import Recipe, RouterDecision
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


class ChatbotEngine:
    def __init__(self, config: EngineConfig, deps: EngineDeps) -> None:
        self._config = config
        self._deps = deps

    def answer(
        self, message: str, *, session_id: str, session_state: SessionState
    ) -> tuple[BotResponse, SessionState]:
        trace_id = uuid4().hex

        new_state = session_state  # By default, the state doesn't change. Recipes can override this if needed.

        # Trace: engine start
        log_event(
            trace_id=trace_id,
            event="engine.start",
            payload={
                "session_id": session_id,
                "message": message,
                "app": self._config.app_name,
            },
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

                    query_result: TxQAQueryResult = TxQAQueryResult(
                        query=parsed_query,
                        total=total_spent,
                        created_at=datetime.now(
                            timezone.utc
                        ),  # Using current UTC time as a timestamp
                    )

                    # Update state with the new query result
                    new_state = replace(
                        session_state,
                        txs_results=session_state.txs_results + (query_result,),
                    )

                    answer_text = f"You spent a total of {total_spent:.2f} EUR on {parsed_query.label} from {parsed_query.start} to {parsed_query.end}.\n"

            case Recipe.TX_EXPLAIN:
                txs_results = session_state.txs_results
                last_txs_result = txs_results[-1] if txs_results else None

                if last_txs_result is None:
                    answer_text = "Sorry, I don't have any transaction summary to explain. Please ask a question about your spending first (e.g., 'How much did I spend on food last month?').\n"
                else:
                    answer_text = f"Here are the transactions that contributed to this ({last_txs_result.total:.2f} EUR) sum:\n"
                    txs = self._deps.tx_repository.list_transactions(
                        TxFilter(
                            label=last_txs_result.query.label,
                            start=last_txs_result.query.start,
                            end=last_txs_result.query.end,
                            direction="spend",
                        )
                    )
                    for tx in txs:
                        answer_text += f"- {tx.date}: {tx.amount:.2f} EUR to {tx.other_account} ({tx.description})\n"

            case Recipe.DOCS_ANSWER:
                answer_text = "DOCS_ANSWER is not implemented yet.\n"
            case Recipe.OUT_OF_SCOPE:
                answer_text = "Sorry, I can't help with that.\n"
            case _:
                answer_text = "Unknown recipe.\n"

        bot_response = BotResponse(answer=answer_text, references=[], trace_id=trace_id)
        result: tuple[BotResponse, SessionState] = (bot_response, new_state)

        # Trace: engine finish
        log_event(
            trace_id=trace_id,
            event="engine.finish",
            payload={"references": bot_response.references},
        )

        return result
