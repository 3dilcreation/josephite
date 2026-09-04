import { db } from "@/lib/db";
import type { NotificationType, EntityType } from "@prisma/client";

type NotifyInput = {
  orgId: string;
  userIds: (string | null | undefined)[];
  type: NotificationType;
  title: string;
  body?: string;
  link?: string;
  /** Don't notify the person who caused the event. */
  exceptUserId?: string;
};

export async function notify({ orgId, userIds, type, title, body, link, exceptUserId }: NotifyInput) {
  const targets = [...new Set(userIds.filter((id): id is string => !!id))].filter(
    (id) => id !== exceptUserId,
  );
  if (targets.length === 0) return;

  await db.notification.createMany({
    data: targets.map((userId) => ({ orgId, userId, type, title, body, link })),
  });
}

export async function logActivity(input: {
  orgId: string;
  entityType: EntityType;
  entityId: string;
  action: string;
  summary: string;
  actorId?: string | null;
  meta?: Record<string, unknown>;
}) {
  await db.activity.create({
    data: {
      orgId: input.orgId,
      entityType: input.entityType,
      entityId: input.entityId,
      action: input.action,
      summary: input.summary,
      actorId: input.actorId ?? null,
      meta: input.meta as never,
    },
  });
}
