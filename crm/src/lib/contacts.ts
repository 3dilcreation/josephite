import type { Prisma } from "@prisma/client";

/**
 * Build the OR clause that matches an existing customer by phone or email.
 * Returns null when we have neither, so callers can skip the lookup instead of
 * running a query that would match every row.
 */
export function contactMatch(input: {
  phone?: string | null;
  email?: string | null;
}): Prisma.CustomerWhereInput[] | null {
  const clauses: Prisma.CustomerWhereInput[] = [];
  if (input.phone) clauses.push({ phone: input.phone });
  if (input.email) clauses.push({ email: input.email });
  return clauses.length > 0 ? clauses : null;
}
