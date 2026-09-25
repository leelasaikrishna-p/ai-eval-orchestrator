# Ledgerly Help Center (sample knowledge base)

Ledgerly is a fictional invoicing tool for small businesses, used here only
as a small, self-contained domain for evaluating a RAG-based FAQ bot. None
of this is real product documentation.

## Invoicing
Invoices in Ledgerly can be sent as email or as a shareable link. Every
invoice supports partial payments -- a customer can pay any amount up to
the invoice total, and the invoice stays open until the full balance is
paid. Invoices can be edited any time before the first payment is received;
after a partial payment, only the due date and notes fields remain editable.

## Payment terms
Default payment terms are Net 30. Account owners can change the default to
Net 15, Net 45, or Net 60 in Settings > Billing. Changing the default only
affects invoices created after the change -- existing invoices keep their
original terms.

## Refunds
Refunds can be issued for any payment within 180 days of the payment date.
A refund can be partial or full. Refunded amounts are returned to the
original payment method and typically post within 5-10 business days.
Ledgerly does not support refunding a payment to a different payment method
than the one used originally.

## Late fees
Late fees are off by default. When enabled in Settings > Billing, a late
fee (flat amount or percentage) is applied automatically 1 day after an
invoice's due date, and again every 30 days the invoice remains unpaid, up
to a cap the account owner sets.

## Multi-currency
Ledgerly supports invoicing in USD, EUR, GBP, and CAD. An account's default
currency is set once at account creation and cannot be changed afterward,
though individual invoices can still be issued in any of the four supported
currencies regardless of the account default.
