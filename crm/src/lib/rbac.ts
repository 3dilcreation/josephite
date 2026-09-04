import type { Prisma, Role } from "@prisma/client";
import type { CurrentUser } from "@/lib/auth";

// Visibility rules, in one place, so "who can see this" never drifts between
// the list page, the detail page and the export.
//
//   SUPER_ADMIN  every branch in the organisation
//   ADMIN        their own branch (plus rows not yet assigned to one)
//   MANAGER      their departments, or work assigned to them
//   STAFF        their departments, or work assigned to them
//   VIEWER       same read scope as ADMIN, but no write anywhere

export function canManageUsers(role: Role) {
  return role === "SUPER_ADMIN" || role === "ADMIN";
}

export function canManageOrg(role: Role) {
  return role === "SUPER_ADMIN";
}

export function canWrite(role: Role) {
  return role !== "VIEWER";
}

export function canReassign(role: Role) {
  return role === "SUPER_ADMIN" || role === "ADMIN" || role === "MANAGER";
}

export function canSeeAllBranches(role: Role) {
  return role === "SUPER_ADMIN";
}

/**
 * Rows belonging to this user's branch, plus rows with no branch set yet (an
 * inbound lead from a channel with no default branch, for instance) — otherwise
 * unrouted work would be invisible to the very people meant to route it.
 *
 * Expressed as an OR rather than `branchId: { in: [id, null] }`, which Prisma
 * rejects: `in` takes a list of values, and null is not one of them.
 */
function branchClause(branchId: string | null): Prisma.OrderWhereInput {
  if (!branchId) return {};
  return { OR: [{ branchId }, { branchId: null }] };
}

/**
 * Prisma `where` fragment limiting Lead/Order rows to what this user may see.
 * Callers must combine it under an `AND`, never by spreading it — spreading
 * would let a page's own `OR` (a search box, say) silently replace the scope.
 */
export function scopeWhere(user: CurrentUser): Prisma.OrderWhereInput {
  const base: Prisma.OrderWhereInput = { orgId: user.orgId };

  if (user.role === "SUPER_ADMIN") return base;

  if (user.role === "ADMIN" || user.role === "VIEWER") {
    return { ...base, ...branchClause(user.branchId) };
  }

  // MANAGER and STAFF: department work plus anything pointed at them directly.
  return {
    ...base,
    OR: [
      { assignedToId: user.id },
      user.departmentIds.length > 0
        ? { departmentId: { in: user.departmentIds } }
        : { departmentId: "__none__" },
    ],
  };
}

/** Same idea for customers, which are owned rather than assigned. */
export function customerScopeWhere(user: CurrentUser): Prisma.CustomerWhereInput {
  const base: Prisma.CustomerWhereInput = { orgId: user.orgId };
  if (user.role === "SUPER_ADMIN") return base;
  if (user.role === "ADMIN" || user.role === "VIEWER") {
    return { ...base, ...(branchClause(user.branchId) as Prisma.CustomerWhereInput) };
  }
  return base;
}

export const ROLE_LABELS: Record<Role, string> = {
  SUPER_ADMIN: "Super Admin",
  ADMIN: "Branch Admin",
  MANAGER: "Department Manager",
  STAFF: "Staff",
  VIEWER: "Viewer (read only)",
};
