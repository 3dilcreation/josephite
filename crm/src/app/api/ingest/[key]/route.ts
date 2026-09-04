import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { processInboundPayload, verifySignature } from "@/lib/ingest/process";

export const dynamic = "force-dynamic";

/**
 * Single inbound endpoint for every channel:
 *   POST /api/ingest/<source key>
 *   x-3dil-signature: sha256=<hmac of the raw body using the source secret>
 *
 * The raw body is stored before it is interpreted, so a mapping bug can be
 * fixed and the event replayed rather than losing a customer enquiry.
 */
export async function POST(request: Request, context: { params: Promise<{ key: string }> }) {
  const { key } = await context.params;
  const raw = await request.text();

  const source = await db.integrationSource.findFirst({ where: { key } });
  if (!source || !source.isActive) {
    return NextResponse.json({ error: "Unknown or inactive source" }, { status: 404 });
  }

  if (!verifySignature(source.secret, raw, request.headers.get("x-3dil-signature"))) {
    return NextResponse.json({ error: "Invalid signature" }, { status: 401 });
  }

  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(raw);
  } catch {
    return NextResponse.json({ error: "Body must be JSON" }, { status: 400 });
  }

  const event = await db.webhookEvent.create({
    data: { sourceId: source.id, payload: payload as never },
  });

  try {
    const result = await processInboundPayload(source, payload);
    await db.webhookEvent.update({
      where: { id: event.id },
      data: {
        status: result.duplicate ? "IGNORED" : "PROCESSED",
        resultType: result.entityType,
        resultId: result.entityId,
        processedAt: new Date(),
      },
    });
    await db.integrationSource.update({
      where: { id: source.id },
      data: { lastEventAt: new Date() },
    });
    return NextResponse.json({
      ok: true,
      duplicate: result.duplicate,
      type: result.entityType,
      id: result.entityId,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await db.webhookEvent.update({
      where: { id: event.id },
      data: { status: "FAILED", error: message, processedAt: new Date() },
    });
    // 500 so the sending platform retries; the raw event is already safe.
    return NextResponse.json({ error: "Processing failed", detail: message }, { status: 500 });
  }
}

/** Meta platforms verify a webhook with a GET challenge before they will send. */
export async function GET(request: Request) {
  const url = new URL(request.url);
  const challenge = url.searchParams.get("hub.challenge");
  if (challenge) return new Response(challenge, { status: 200 });
  return NextResponse.json({ ok: true });
}
