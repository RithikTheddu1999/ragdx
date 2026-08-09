---
title: Hazardous materials handling
department: logistics
status: current
---
Dangerous goods move on the Northwind network only where the origin depot, the
destination depot and every linehaul leg between them are certified for the
relevant class. The routing engine enforces this automatically, but a manual
rebooking by the support team can bypass the check, which is why manual
rebooking of a dangerous goods shipment requires a duty manager code.

Every dangerous goods consignment needs a shipper's declaration, a class label
on two adjacent faces, and an emergency contact number staffed twenty four hours
a day. A declaration missing the emergency contact is rejected at induction and
the shipment is returned to the shipper at the shipper's cost.

Class | Description | Packing group permitted | Depot certification required
Class 2.1 | Flammable gas | not applicable | full dangerous goods depot
Class 3 | Flammable liquid | I, II and III | full dangerous goods depot
Class 6.1 | Toxic substance | I, II and III | full dangerous goods depot
Class 9 | Miscellaneous, including lithium batteries | II and III only | limited quantity depot
Class 8 | Corrosive substance | II and III only | limited quantity depot

Limited quantity shipments are exempt from the full declaration requirement but
still require the class label and the emergency contact. The exemption applies
per package, not per consignment, so a pallet of limited quantity packages is
still a limited quantity shipment.

A spill, a leak or a damaged package containing dangerous goods is a category one
incident. The depot must isolate the package, evacuate the immediate area and
page the regional safety lead within five minutes. The escalation window for a
category one incident does not pause overnight or at weekends.

Training expires after twenty four months. A depot whose certified handler count
falls below two loses its certification at the end of the month and the routing
engine will stop offering dangerous goods service from that site.
