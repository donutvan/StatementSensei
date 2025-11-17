from dataclasses import dataclass
from datetime import date

from monopoly.statements import Transaction


@dataclass
class TransactionMetadata:
    bank_name: str
    account_number: str | None = None
    currency: str | None = None
    statement_start: date | None = None
    statement_end: date | None = None


@dataclass
class ProcessedFile:
    transactions: list[Transaction]
    metadata: TransactionMetadata

    def __iter__(self):
        return iter(self.transactions)
