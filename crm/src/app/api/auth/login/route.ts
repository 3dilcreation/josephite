import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { z } from "zod";
import { db } from "@/lib/db";
import { SESSION_COOKIE, encodeSession, sessionCookieOptions } from "@/lib/session";

const schema = z.object({ email: z.string().email(), password: z.string().min(1) });

export async function POST(request: Request) {
  const parsed = schema.safeParse(await request.json().catch(() => ({})));
  if (!parsed.success) {
    return NextResponse.json({ error: "Enter a valid email and password" }, { status: 400 });
  }

  const user = await db.user.findUnique({
    where: { email: parsed.data.email.toLowerCase() },
  });

  // Same message and roughly the same work either way, so the response does not
  // reveal which accounts exist.
  const hash = user?.passwordHash ?? "$2a$10$invalidinvalidinvalidinvalidinvalidinvalidinvalidinvalidinv";
  const valid = await bcrypt.compare(parsed.data.password, hash);

  if (!user || !valid || !user.isActive) {
    return NextResponse.json({ error: "Incorrect email or password" }, { status: 401 });
  }

  const token = await encodeSession({
    userId: user.id,
    orgId: user.orgId,
    branchId: user.branchId,
    role: user.role,
    name: user.name,
    email: user.email,
  });

  await db.user.update({ where: { id: user.id }, data: { lastLoginAt: new Date() } });

  const response = NextResponse.json({ ok: true });
  response.cookies.set(SESSION_COOKIE, token, sessionCookieOptions);
  return response;
}
