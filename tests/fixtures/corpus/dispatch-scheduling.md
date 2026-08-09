---
title: Dispatch scheduling
department: logistics
status: current
---
Dispatch scheduling runs three waves per day at every Northwind depot. The wave
plan determines which shipments leave the building and in what order the loading
docks are worked. Wave planning is owned by the depot duty manager and is not
something the support team may change on a customer's behalf.

The morning wave is built at 05:30 and covers everything that arrived overnight
by linehaul. The midday wave is built at 12:15 and is the wave that absorbs
same-day pickups from the metro area. The evening wave is built at 19:45 and is
the last chance for a shipment to move on the current transit day.

Wave cutoffs are not the same as the published order cutoff for a service tier.
The order cutoff is a commercial promise to the customer; the wave cutoff is an
operational deadline inside the depot. A shipment can meet the order cutoff and
still miss the wave if it is not physically at the outbound dock in time, which
is why the operations team tracks dock arrival separately.

Priority Air consignments are always worked first within a wave, then Regional
Express, then Standard Ground. Freight Consolidated is worked last and may be
rolled to the next wave without an incident being raised, because consolidation
is explicitly a best-effort transit product.

If a wave is running more than forty minutes behind, the duty manager must post
a delay notice to the depot channel and page the regional operations lead. A
delay notice is not an incident code; it is an internal signal that lets the
support team set customer expectations before the complaints arrive.

Depots that consistently miss the evening wave are reviewed monthly. The review
looks at dock arrival times, staffing on the outbound shift, and whether the
depot is accepting late tenders it has no realistic chance of moving.
