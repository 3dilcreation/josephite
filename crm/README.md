# 3DIL CREATION — CRM

An operations CRM for a 3D printing business: leads, orders, production stages,
payments, departments and multi-channel intake, in one place.

Built as a multi-tenant application from the first commit. Running a single
location today costs nothing extra; adding a second branch or a franchisee later
is a row in a table, not a rewrite.

## Stack

| Layer     | Choice                                        |
| --------- | --------------------------------------------- |
| Framework | Next.js 15 (App Router, React Server Components) |
| Language  | TypeScript                                    |
| Database  | PostgreSQL via Prisma                         |
| Auth      | HMAC-signed JWT in an httpOnly cookie, bcrypt password hashes |
| Styling   | Tailwind CSS v4                               |

## Running it locally

```bash
cp .env.example .env      # set DATABASE_URL and a 32+ char AUTH_SECRET
npm install
npx prisma db push        # create the schema
npm run db:seed           # departments, demo users, sample data
npm run dev               # http://localhost:3000
```

The seed prints its credentials. Defaults are `admin@3dilcreation.com` /
`ChangeMe123!` — override with `SEED_ADMIN_EMAIL` and `SEED_ADMIN_PASSWORD`, and
change the password immediately on any deployment you can reach from the internet.

## Roles and what each one sees

Visibility is enforced in the database query, not just hidden in the UI — see
`src/lib/rbac.ts`, which every list and detail page routes through.

| Role            | Scope                                                        |
| --------------- | ------------------------------------------------------------ |
| `SUPER_ADMIN`   | Every branch, plus users, branches and integrations           |
| `ADMIN`         | Their own branch, and can manage users within it              |
| `MANAGER`       | Their departments' work, and can reassign inside them         |
| `STAFF`         | Work assigned to them or to their departments                 |
| `VIEWER`        | Read-only across the branch — for an accountant or auditor    |

A branch admin cannot create or edit a super admin, and nobody can deactivate
their own account. Both rules live in the server action, so they hold whatever
the browser sends.

## Departments

Seeded with Sales & Marketing, Design, Production, Dispatch & Support, and
Accounts. A person can belong to more than one — common in a small shop where the
same person quotes in the morning and runs post-processing in the afternoon.

Leads and orders are routed to a department, so work stays visible to a team
rather than trapped with one individual who is on leave.

## Order lifecycle

`DRAFT → CONFIRMED → DESIGN → PRINTING → POST_PROCESSING → QUALITY_CHECK →
READY → SHIPPED → DELIVERED`, with `ON_HOLD` and `CANCELLED` available at any
point. The production board at `/production` shows every live job in these
columns, urgent first, then earliest deadline.

3D-printing specifics live on the **line item**, not the order, because one order
routinely mixes an SLA resin prototype with a batch of FDM jigs: technology
(FDM/SLA/SLS/MJF/DMLS/PolyJet/DLP), material, colour, weight in grams, print
hours, layer height, infill and the model file name.

### Money

Order totals are always **derived** from line items and recorded payments
(`recalculateOrder` in `src/lib/orders.ts`), never taken from the form. Payment
status follows from the amounts and the payment due date, so `PARTIAL` and
`OVERDUE` cannot drift out of sync with reality.

## Channel integrations

Every channel posts to one endpoint:

```
POST /api/ingest/<source key>
x-3dil-signature: sha256=<HMAC-SHA256 of the raw JSON body, keyed with the source secret>
```

Create a source at **Admin → Integrations** to get its URL and secret. The raw
payload is stored in `WebhookEvent` *before* it is interpreted, so a bad field
mapping can be fixed and the event replayed rather than losing a customer enquiry.

Adapters (`src/lib/ingest/normalize.ts`) flatten each platform's JSON into either
a normalised lead or a normalised order. Currently handled:

- **Website / generic form** — `name`, `email`, `phone`, `message`, `budget`
- **IndiaMART** — its `SENDER_*` push fields; enquiries with a product category
  are raised to HIGH priority
- **WhatsApp, Instagram, Facebook lead ads** — flat fields or Meta's `field_data`
  envelope, plus the `hub.challenge` GET verification Meta requires
- **Shopify / WooCommerce** — full order with line items; a `paid` order records
  its payment automatically
- **Amazon, Flipkart, Etsy** — the common order fields across their report shapes

Adding another marketplace is one mapper function plus a `SourceKind` value.
Nothing downstream changes.

Two protections that matter in production: inbound events are **deduplicated** on
`(source, external id)`, so a platform retry cannot create the same lead twice;
and customers are **merged by phone or email**, so the same person ordering from
your website and from Amazon stays one record with one order history.

### Sending a test payload

```bash
SECRET=<from Admin → Integrations>
BODY='{"name":"Test","phone":"9999999999","message":"20 PLA brackets"}'
SIG=$(node -e "const c=require('crypto');process.stdout.write(c.createHmac('sha256',process.argv[1]).update(process.argv[2]).digest('hex'))" "$SECRET" "$BODY")
curl -X POST http://localhost:3000/api/ingest/<key> -H "x-3dil-signature: sha256=$SIG" -d "$BODY"
```

## Notifications

In-app, generated on assignment, order status moves, and payments. Notifying the
department (rather than one person) when an inbound lead has no owner yet means
enquiries are not silently parked. The person who caused an event is never
notified about their own action.

Email, WhatsApp and push delivery are not built — see the roadmap.

## Audit trail

Every meaningful change writes to `Activity`: who changed a status, reassigned a
job or recorded a payment, and when. It is append-only and shown on each lead and
order page.

## Deploying

Any Node host with a PostgreSQL database. Before going live:

1. Set `AUTH_SECRET` to a fresh random value (`openssl rand -base64 32`).
2. Set `APP_URL` to your real domain — integration endpoints are built from it.
3. Change the seeded passwords.
4. Serve over HTTPS. The session cookie sets `Secure` in production, and the
   webhook signature check is only meaningful over TLS.
5. Use `prisma migrate deploy` rather than `db push` once you have real data.

## What is deliberately not here

Read `ROADMAP.md` for the full list with reasoning. The short version: staff
attendance, e-invoicing, and outbound WhatsApp were scoped out of this build.
