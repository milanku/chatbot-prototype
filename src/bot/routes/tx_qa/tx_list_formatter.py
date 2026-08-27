from bot.models.tx_qa.domain import Transaction


def format_transactions(txs: list[Transaction]) -> str:
        if not txs:
            return "No transactions found for the specified query."

        lines = ["Here are your transactions:"]

        for tx in txs:
            lines.append(
                f"- {tx.date}: {tx.amount:.2f} EUR "
                f"to {tx.other_account} ({tx.description})"
            )

        return "\n".join(lines)