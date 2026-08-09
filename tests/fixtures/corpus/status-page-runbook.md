---
title: Status page runbook
department: engineering
status: current
---
The public status page is the first thing a customer checks and the last thing an
engineer remembers to update. This runbook exists so that updating it is a step
in the incident process rather than an afterthought.

A status page entry is opened for any severity one incident and for any severity
two incident lasting more than thirty minutes. The incident commander opens the
entry; the owning team does not, because an engineer mid diagnosis should not be
writing customer copy.

The first entry goes up within ten minutes of the incident being acknowledged
and says only what is known: which component, which customer visible symptom, and
when the next update will come. It never contains a cause, because the first
cause an engineer proposes is wrong often enough to be dangerous in public.

Updates follow at the interval promised, even when there is nothing new. An
update that says investigation continues and repeats the next update time is
better than silence, and silence is what generates the support contact volume
that swamps the escalation queue.

Component names on the status page must match the names customers see in the
portal. Internal service names never appear. The mapping is maintained by the
platform support team and reviewed whenever a component is renamed.

Resolution requires two things: the customer visible symptom is gone, and the
owning team has confirmed it. A status page entry closed on a monitoring signal
alone is reopened often enough that the incident commander is expected to wait
for the human confirmation.

A post incident review is published for every severity one within five business
days. The review names contributing factors and follow up actions with owners.
It does not name individuals, and a review that reads as a search for someone to
blame is sent back by the duty director.
