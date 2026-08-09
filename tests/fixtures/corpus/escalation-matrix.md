---
title: Escalation matrix
department: customer-support
status: current
---
The escalation matrix tells a support agent which team owns an incident and how
long the escalation window is before the incident moves up a level. Agents
should consult the matrix before contacting a team directly, because a direct
contact that bypasses the matrix does not stop the escalation clock.

Severity one covers anything that stops shipments moving across a region or
exposes customer data. The owning team must acknowledge within fifteen minutes at
any hour. The escalation window does not pause overnight, at weekends or during
a public holiday.

Severity two covers a single depot, a single large customer account, or a
platform component degraded but working. The owning team acknowledges within one
hour during business hours and within four hours outside them.

Severity three covers everything else, including individual shipment queries
that need a specialist team. The escalation window is one business day.

Team | Owns | Business hours window | Out of hours window
Network control | linehaul, wave plans, depot capacity | 30 minutes | 2 hours
Platform support | scanners, handhelds, portal, address services | 1 hour | 4 hours
Regional safety | dangerous goods, spills, injuries | 15 minutes | 15 minutes
Claims | damage, loss, commercial disputes | 1 business day | none
Identity | login, single sign on, API credentials | 1 hour | 4 hours

An agent may raise the severity of an incident but may not lower it. Lowering
severity is a duty manager action and must be recorded with a reason, because a
silently downgraded incident is how a small problem becomes a customer
escalation two days later.

When an incident code and the matrix disagree about the owning team, the incident
code wins and the agent should raise a correction against the matrix. The matrix
is reviewed monthly; the incident code list is reviewed whenever a code changes.
