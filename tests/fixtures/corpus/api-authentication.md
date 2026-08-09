---
title: API authentication
department: engineering
status: current
---
The Northwind shipping API authenticates with OAuth 2.0 client credentials. Long
lived static keys were retired and any integration still presenting one receives
a 401 with an explanatory body rather than a silent failure.

A client obtains an access token by posting its client identifier and client
secret to the token endpoint. The token is valid for one hour. Clients should
refresh on expiry rather than on a fixed timer, because the platform occasionally
issues shorter lived tokens during a credential rotation.

Client secrets are issued through the developer portal and are shown exactly
once. A lost secret cannot be recovered and must be rotated. Rotation creates a
second active secret so an integration can move across without downtime; the old
secret is revoked automatically after fourteen days.

Every request carries the token in an Authorization header with the Bearer
scheme. Requests that also present a legacy key header are rejected outright
rather than falling back, because a silent fallback is how a customer keeps
running on a credential they believe they have revoked.

Scopes are coarse. The shipments scope permits booking, tracking and label
generation. The rates scope permits tariff queries only. The admin scope permits
credential management and is never granted to an integration that also books
shipments.

Repeated authentication failures from one client are raised as an incident with
code NW-4440 and routed to the identity team. The escalation window is one hour
during business hours. Support agents should not attempt to reset a customer
credential themselves; only the identity team may do so, and only after
verifying the caller through the account contact on record.

Sandbox credentials are entirely separate and never work against production. A
team that has been testing successfully for weeks and fails on the first
production call has almost always kept the sandbox client identifier.
