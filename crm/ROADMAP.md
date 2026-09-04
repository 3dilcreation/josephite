# What to build next — and what to skip

Honest ratings out of 10. The score is **return on effort for a 3D printing
service business**, not how impressive the feature sounds in a pitch. Anything
below 5 is something I would actively argue against building right now.

Three things about the original brief, before the list:

1. **"Integration with all the sites" is not a feature, it is N projects.**
   The intake plumbing is built and adapters exist for Shopify/Woo, Amazon,
   Flipkart, Etsy, IndiaMART, WhatsApp, Instagram and Facebook lead ads. But each
   platform still needs real API credentials, its own OAuth or token flow, its own
   sandbox testing and its own rate-limit handling. Budget 1–3 weeks per
   marketplace. Do IndiaMART and your own website first — for an Indian
   manufacturing business those two will out-produce the rest combined.

2. **Franchise support before you have franchisees is premature**, and you said
   you are one location today. The data model carries `Branch` and a royalty
   percentage so the door is open, but do not spend weeks on franchise billing
   flows until a franchisee actually exists and tells you how they want to be
   billed. You will guess wrong.

3. **GPS attendance does not belong in a CRM.** You have already deferred it,
   which was the right call. When you do want it: browser geolocation is trivially
   spoofed (a developer-tools override, or any of a dozen free Android mock-location
   apps), so an honest web check-in is a *log*, not *proof*. Real enforcement needs
   a native app with mock-location detection. At under 20 staff on one site, a
   ₹200/month attendance app will beat anything I build you here.

---

## Tier 1 — build these next

### Instant file-based quoting · 9/10
Customer uploads an STL or STEP; the system computes bounding box, volume and
estimated print time, then prices it from your material and machine rates and
returns a quote in seconds.

This is the single highest-leverage thing you can add. It converts the "send us
your file and we'll get back to you" delay — where most 3D printing enquiries die —
into an instant number. It also kills the hours your sales person currently spends
quoting jobs that were never going to close. Everything else on this list improves
an existing process; this one changes your conversion rate.

Not trivial: volume from a mesh is easy, *print time* is not, and a naive estimate
will be wrong enough to lose you money. Start by pricing on volume plus bounding
box with a manual review step above a value threshold, and calibrate against real
slicer output over a few months.

### True job costing and per-order margin · 9/10
Machine hours × machine rate + material grams × material cost + labour +
post-processing + an allowance for failed prints. Show gross margin on every order
and per customer.

You are probably already carrying at least one customer who feels like a good
account and is actually unprofitable once failed prints and rework are counted.
The schema already stores `weightGrams` and `printHours` per line item; this is
mostly rate tables and reporting on data you are collecting.

### SLA timers and automatic escalation · 8/10
"No contact within 4 hours" → notify the manager. "Order due in 24 hours and still
in Design" → escalate. "Payment overdue by 7 days" → chase.

Cheap to build on the existing notification and activity tables — a scheduled job
and a rules table. This is what stops leads quietly rotting, which is the failure
mode of every CRM that people forget to open.

### WhatsApp outbound updates · 8/10
"Your order is printing", "ready for pickup", "payment received". In India this is
the channel customers actually read. Inbound is already handled; outbound needs a
WhatsApp Business API provider and pre-approved message templates.

Cuts "where is my order?" calls sharply, and every one of those calls currently
interrupts someone on the shop floor.

### Material inventory · 8/10
Spool and resin bottle tracking with gram-level consumption deducted as jobs
complete, plus low-stock alerts and per-batch cost.

You cannot cost a job honestly without it, and running out of a specific filament
mid-batch is a deadline-missing event. Pairs directly with job costing.

### Machine registry and capacity planning · 8/10
Every printer as a record with technology, build volume, hourly rate and status.
Assign jobs to machines; see utilisation and where the queue is jammed.

Answers "can we take this rush job?" with a fact instead of a guess, and tells you
which printer is actually earning its keep before you buy another one.

### GST invoicing and e-invoicing · 8/10
Invoice PDFs with your GSTIN, HSN codes, correct CGST/SGST/IGST split, and IRN
generation via an IRP if you cross the e-invoicing turnover threshold.

Not exciting, but it is compliance, and doing it in the CRM removes a whole
duplicate-entry step into Tally or Zoho.

### Reports that answer real questions · 8/10
Source-wise conversion and revenue (which channel actually pays for itself),
lead-to-order conversion by owner, average turnaround by technology, machine
utilisation, repeat-customer rate, print failure rate by material and machine.

The last one especially: failure rate by material and machine is where a printing
business silently loses money.

---

## Tier 2 — clearly worth doing, but after Tier 1

### Customer portal · 7/10
Customers log in to see order status, approve a quote, upload revised files and
download invoices. Removes status-chasing calls and makes you look considerably
larger than you are. Build it after the quoting engine — the portal is much more
useful when it can quote.

### Quote documents with online approval · 7/10
Versioned quote PDFs with a click-to-approve link and an audit trail. Removes the
"which version did they agree to?" argument, which becomes expensive exactly once.

### QR job travellers · 7/10
Print a QR sticker per job. Operators scan it to advance the stage from their
phone instead of walking to a computer. This is what makes shop-floor staff
actually keep the CRM current — and stale data is what kills internal tools.

### File storage with versioning and 3D preview · 7/10
Model files attached to orders, in S3 or similar, with revisions kept and an
in-browser STL preview. Today the app stores a file *name*; that is a deliberate
placeholder. Stop emailing `final_v2_FINAL.stl`.

### Recurring and blanket orders · 6/10
Standing monthly quantities for B2B accounts, auto-generating each cycle. Worth
it once you have three or four repeat industrial customers; not before.

### Mobile PWA polish · 6/10
The app is already responsive. Installable-with-offline-read is the next step, and
matters most for the production board and the QR flow above.

---

## Tier 3 — be sceptical

### AI features · 5/10 *as usually pitched*
Lead scoring, auto-drafted quote emails, chat summarisation. Fine, but at your
volume a human reading the enquiry is fast and more accurate, and a lead-scoring
model trained on a few hundred rows is noise with a confidence interval.

The genuinely useful AI application here is narrower: **parsing an incoming
enquiry into structured fields** (quantity, material, deadline) so leads arrive
pre-filled, and **checking uploaded meshes for printability** — thin walls, non-
manifold geometry, unsupported overhangs — before you quote. Those are worth
7/10. The generic "AI CRM assistant" is not.

### Email and calendar sync · 5/10
Two-way Gmail threading against customer records. Nice, fiddly, and mostly
duplicates what WhatsApp already does for your customer base.

### Attendance with GPS · 4/10 *in this system*
Covered above. Buy it, do not build it.

### Multi-currency and export documentation · 4/10
Real if you ship internationally, dead weight if you do not. The schema already
carries a currency per order, so this stays cheap to add later.

### Full accounting · 3/10
Do not build a ledger. Integrate with Tally, Zoho Books or QuickBooks when the
invoicing above is in place.

---

## Suggested sequence

**Next 4–6 weeks** — file-based quoting (with manual review above a threshold),
material inventory, job costing on top of both. These three compound: quoting
needs material costs, costing needs consumption data.

**Following 4–6 weeks** — SLA escalation, WhatsApp outbound, GST invoicing. Each
is independent, so they can run in parallel or be dropped without blocking
anything else.

**Then** — machine registry and capacity planning, the reporting suite, and the
customer portal, in that order.

**Only when a franchisee actually exists** — franchise billing, royalty
reconciliation and inter-branch inventory transfer.

## One thing to watch

The largest risk to this system is not a missing feature. It is that shop-floor
staff stop updating it, at which point every number on the dashboard becomes a
lie and people go back to WhatsApp groups and a whiteboard.

Everything that reduces the friction of keeping data current — QR job travellers,
the mobile board, WhatsApp updates that save someone a phone call — is worth more
than it looks on a feature list. Weigh those above anything that only produces a
prettier report.
