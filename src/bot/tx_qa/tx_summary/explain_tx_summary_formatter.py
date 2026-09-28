from bot.tx_qa.tx_summary.models import SummaryQueryResult


def format_tx_explain_summary(related_summaries: list[SummaryQueryResult]) -> str:
    answer_text = "Here are the transactions that contributed to your selected sum:\n"
    for tx_result in related_summaries:
        for tx in tx_result.transactions:
            answer_text += (
                f"- {tx.date}: {tx.amount:.2f} EUR to {tx.other_account} ({tx.description})\n"
            )
    return answer_text
