---
title: Operational incident codes
department: engineering
status: current
---
Every operational incident raised on the Northwind platform carries an incident
code. The code drives routing to the owning team, the escalation window, and
whether the incident appears on the executive incident report. Support agents
must not invent a code; if no code fits, the incident is raised as NW-4400 and
the triage team assigns a real code within one hour.

An incident code is not a delay notice and not a damage claim. A delay notice is
an internal signal posted by a depot duty manager. A damage claim is a
commercial process owned by the claims team. Raising an incident code for either
of these creates duplicate work and distorts the incident volume the operations
team reviews each week.

The escalation window attached to each code is the time the owning team has to
acknowledge, not to resolve. A missed acknowledgement escalates automatically to
the regional lead, and a second miss escalates to the duty director.

NW-4400 | Unclassified operational incident awaiting triage | route to triage
NW-4405 | Linehaul departed with unscanned freight on board | route to network control
NW-4410 | Barcode scanner offline at induction | route to platform support
NW-4417 | Reefer door ajar for more than five minutes | isolate the unit, then page the duty officer within ten minutes
NW-4422 | Address validation service returning stale results | route to platform support
NW-4431 | Driver handheld unable to sync at end of shift | route to field systems
NW-4440 | Customer portal login failures above baseline | route to identity
NW-4455 | Dangerous goods package found undeclared | route to regional safety

Codes in the NW-44xx range are operational. Codes in the NW-45xx range are
commercial and are owned by the account management team rather than by
engineering. The two ranges have different escalation windows and different
weekend behaviour, which is a frequent source of confusion for new agents.

Retired codes are never reused. A code that disappears from this list has been
retired and any incident still open against it is migrated by the triage team.
