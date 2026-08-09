---
title: Cold chain operations
department: operations
status: current
---
Cold chain work is the most tightly controlled activity on the network. A single
unmonitored gap invalidates the whole movement, and unlike a damaged parcel there
is no way to inspect the goods afterwards and decide they are fine.

Reefer units are pre-cooled for a minimum of ninety minutes before loading. A
unit that has not reached its target band before the doors open is taken out of
the wave and the shipment is held. Depots may not load into a warm unit on the
promise that it will pull down during transit, because it will not.

Biologic loads travel in reefer units held between 2 and 8 degrees celsius for the
whole journey, and any excursion beyond thirty minutes voids the release
certificate issued at origin.

Frozen loads are held at minus 18 degrees or below. Ambient controlled loads are
held between 15 and 25 degrees. The three bands never share a unit, even when
the volume would allow it, because a mixed load has no defensible audit trail.

Every reefer unit carries two independent data loggers. The primary logger feeds
the telematics platform in near real time; the secondary logger is a sealed
device read out at destination. Where the two disagree by more than one degree
the sealed device is authoritative and an incident is raised against the
telematics vendor.

Door events are the dominant cause of excursions. The telematics platform emits
a door open event with the unit identifier and a timestamp, and the duty officer
is expected to acknowledge it before the escalation window closes. A door left
ajar during a driver break is the single most common finding in cold chain
incident reviews across the network.

Release certificates are generated automatically at destination once the logger
readout is uploaded. A shipment with no certificate cannot be released to the
consignee even when the goods appear to be in perfect condition.
