import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { cache } from "react";
import { db } from "@/lib/db";
import { SESSION_COOKIE, decodeSession } from "@/lib/session";
import type { Role } from "@prisma/client";

export type CurrentUser = {
  id: string;
  orgId: string;
  branchId: string | null;
  role: Role;
  name: string;
  email: string;
  departmentIds: string[];
};

// Cached per request so a page that reads the user in five places hits the
// database once.
export const getCurrentUser = cache(async (): Promise<CurrentUser | null> => {
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!token) return null;

  const payload = await decodeSession(token);
  if (!payload) return null;

  const user = await db.user.findUnique({
    where: { id: payload.userId },
    select: {
      id: true,
      orgId: true,
      branchId: true,
      role: true,
      name: true,
      email: true,
      isActive: true,
      departments: { select: { departmentId: true } },
    },
  });

  // A user deactivated mid-session loses access on their next request rather
  // than when their token happens to expire.
  if (!user || !user.isActive) return null;

  return {
    id: user.id,
    orgId: user.orgId,
    branchId: user.branchId,
    role: user.role,
    name: user.name,
    email: user.email,
    departmentIds: user.departments.map((d) => d.departmentId),
  };
});

export async function requireUser(): Promise<CurrentUser> {
  const user = await getCurrentUser();
  if (!user) redirect("/login");
  return user;
}

export async function requireRole(...roles: Role[]): Promise<CurrentUser> {
  const user = await requireUser();
  if (!roles.includes(user.role)) redirect("/dashboard?denied=1");
  return user;
}
