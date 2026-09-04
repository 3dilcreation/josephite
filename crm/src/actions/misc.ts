"use server";

import { randomBytes } from "node:crypto";
import { revalidatePath } from "next/cache";
import bcrypt from "bcryptjs";
import { db } from "@/lib/db";
import { requireUser, requireRole } from "@/lib/auth";
import { canWrite } from "@/lib/rbac";
import { logActivity, notify } from "@/lib/notify";
import type { EntityType } from "@prisma/client";

const optional = (value: FormDataEntryValue | null) => {
  const text = typeof value === "string" ? value.trim() : "";
  return text === "" ? undefined : text;
};

export async function addNote(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  const entityType = String(formData.get("entityType")) as EntityType;
  const entityId = String(formData.get("entityId"));
  const body = String(formData.get("body") ?? "").trim();
  if (!body) return;

  await db.note.create({ data: { entityType, entityId, body, authorId: user.id } });

  if (entityType === "LEAD") {
    await db.lead.update({ where: { id: entityId }, data: { lastActivityAt: new Date() } });
  }

  revalidatePath(`/${entityType.toLowerCase()}s/${entityId}`);
}

export async function createCustomer(formData: FormData) {
  const user = await requireUser();
  if (!canWrite(user.role)) throw new Error("Read-only access");

  await db.customer.create({
    data: {
      orgId: user.orgId,
      branchId: user.branchId,
      name: String(formData.get("name")),
      company: optional(formData.get("company")),
      email: optional(formData.get("email")),
      phone: optional(formData.get("phone")),
      gstin: optional(formData.get("gstin")),
      address: optional(formData.get("address")),
      city: optional(formData.get("city")),
      state: optional(formData.get("state")),
      pincode: optional(formData.get("pincode")),
      ownerId: user.id,
    },
  });

  revalidatePath("/customers");
}

export async function createUser(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");

  const email = String(formData.get("email")).toLowerCase().trim();
  const password = String(formData.get("password") ?? "");
  if (password.length < 8) throw new Error("Password must be at least 8 characters");

  const existing = await db.user.findUnique({ where: { email } });
  if (existing) throw new Error("A user with that email already exists");

  // A branch admin can only ever create people inside their own branch, and can
  // never mint another super admin.
  const role = String(formData.get("role") ?? "STAFF");
  if (admin.role === "ADMIN" && role === "SUPER_ADMIN") {
    throw new Error("Only a super admin can create another super admin");
  }

  const departmentIds = formData.getAll("departmentIds").map(String).filter(Boolean);

  const user = await db.user.create({
    data: {
      orgId: admin.orgId,
      branchId: admin.role === "ADMIN" ? admin.branchId : optional(formData.get("branchId")),
      email,
      name: String(formData.get("name")),
      phone: optional(formData.get("phone")),
      role: role as never,
      passwordHash: await bcrypt.hash(password, 10),
      departments: { create: departmentIds.map((departmentId) => ({ departmentId })) },
    },
  });

  await logActivity({
    orgId: admin.orgId,
    entityType: "USER",
    entityId: user.id,
    action: "user.created",
    summary: `${admin.name} created ${user.name} (${role})`,
    actorId: admin.id,
  });

  await notify({
    orgId: admin.orgId,
    userIds: [user.id],
    type: "SYSTEM",
    title: "Welcome to the 3DIL CRM",
    body: "Your account has been created. Change your password from your profile.",
    link: "/dashboard",
  });

  revalidatePath("/admin/users");
}

export async function updateUser(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");
  const id = String(formData.get("id"));

  const target = await db.user.findFirst({ where: { id, orgId: admin.orgId } });
  if (!target) throw new Error("User not found");
  if (admin.role === "ADMIN" && target.role === "SUPER_ADMIN") {
    throw new Error("A branch admin cannot modify a super admin");
  }
  // Locking yourself out of your own CRM is a support call nobody enjoys.
  if (target.id === admin.id && formData.get("isActive") === "false") {
    throw new Error("You cannot deactivate your own account");
  }

  const departmentIds = formData.getAll("departmentIds").map(String).filter(Boolean);
  const password = String(formData.get("password") ?? "");

  await db.user.update({
    where: { id },
    data: {
      name: String(formData.get("name") ?? target.name),
      phone: optional(formData.get("phone")),
      role: (formData.get("role") ?? target.role) as never,
      isActive: formData.get("isActive") !== "false",
      ...(password.length >= 8 ? { passwordHash: await bcrypt.hash(password, 10) } : {}),
      departments: {
        deleteMany: {},
        create: departmentIds.map((departmentId) => ({ departmentId })),
      },
    },
  });

  revalidatePath("/admin/users");
}

export async function createDepartment(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");
  const name = String(formData.get("name")).trim();

  await db.department.create({
    data: {
      orgId: admin.orgId,
      name,
      slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""),
      description: optional(formData.get("description")),
      colour: String(formData.get("colour") ?? "#64748b"),
    },
  });

  revalidatePath("/admin/departments");
}

export async function createBranch(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN");

  await db.branch.create({
    data: {
      orgId: admin.orgId,
      name: String(formData.get("name")),
      code: String(formData.get("code")).toUpperCase(),
      type: (formData.get("type") ?? "BRANCH") as never,
      city: optional(formData.get("city")),
      state: optional(formData.get("state")),
      phone: optional(formData.get("phone")),
      email: optional(formData.get("email")),
      royaltyPercent: optional(formData.get("royaltyPercent"))
        ? Number(formData.get("royaltyPercent"))
        : null,
    },
  });

  revalidatePath("/admin/branches");
}

export async function createIntegrationSource(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");
  const name = String(formData.get("name")).trim();

  await db.integrationSource.create({
    data: {
      orgId: admin.orgId,
      name,
      key: `${name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${randomBytes(3).toString("hex")}`,
      kind: (formData.get("kind") ?? "WEBSITE") as never,
      secret: randomBytes(24).toString("hex"),
      defaultDepartmentId: optional(formData.get("defaultDepartmentId")),
      defaultBranchId: optional(formData.get("defaultBranchId")),
    },
  });

  revalidatePath("/admin/integrations");
}

export async function toggleIntegrationSource(formData: FormData) {
  const admin = await requireRole("SUPER_ADMIN", "ADMIN");
  const id = String(formData.get("id"));

  const source = await db.integrationSource.findFirst({ where: { id, orgId: admin.orgId } });
  if (!source) throw new Error("Source not found");

  await db.integrationSource.update({
    where: { id },
    data: { isActive: !source.isActive },
  });

  revalidatePath("/admin/integrations");
}

export async function markNotificationsRead() {
  const user = await requireUser();
  await db.notification.updateMany({
    where: { userId: user.id, readAt: null },
    data: { readAt: new Date() },
  });
  revalidatePath("/notifications");
  revalidatePath("/dashboard");
}
