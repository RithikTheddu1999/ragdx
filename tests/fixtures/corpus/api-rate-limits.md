---
title: API rate limits
department: engineering
status: current
---
Rate limits on the Northwind shipping API are applied per client identifier, not
per source address, so spreading traffic across more machines does not raise the
ceiling for an integration.

The default ceiling is six hundred requests per minute for the shipments scope
and one hundred and twenty requests per minute for the rates scope. Rate quoting
is deliberately tighter because a rate query is far more expensive to serve than
a tracking lookup.

Every response carries the remaining budget and the reset timestamp in headers.
A client that reads those headers and paces itself will never be throttled. A
client that ignores them and retries immediately on a 429 will be throttled
harder, because the platform applies a back off multiplier to clients that retry
inside the reset window.

Burst allowance is twice the per minute ceiling over any ten second interval.
This exists so a batch of label generations at the start of a wave does not fail,
and it is not a licence to run sustained traffic at twice the ceiling.

A raised ceiling is a commercial change, not a support change. The account
manager records the new ceiling in the contract and the identity team applies it
at the next credential rotation. Support agents cannot raise a limit and should
not promise a customer that it will happen the same day.

Sustained throttling of a large account is raised as an incident and routed to
platform support. The escalation window is one hour during business hours and
four hours out of hours, which matches the platform support row in the
escalation matrix.

Webhook delivery is not rate limited inbound but is retried with exponential
back off for up to twenty four hours. A customer endpoint returning 500 for a
full day will have its webhook subscription suspended and an incident raised
against the account.
