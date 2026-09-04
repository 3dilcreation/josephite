import Link from "next/link";
import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { markNotificationsRead } from "@/actions/misc";
import { formatDateTime, humanise } from "@/lib/format";
import { PageHeader, EmptyState } from "@/components/ui";
import { Badge } from "@/components/badges";

export const dynamic = "force-dynamic";

export default async function NotificationsPage() {
  const user = await requireUser();

  const notifications = await db.notification.findMany({
    where: { userId: user.id },
    orderBy: { createdAt: "desc" },
    take: 100,
  });

  const unread = notifications.filter((item) => !item.readAt).length;

  return (
    <>
      <PageHeader
        title="Notifications"
        subtitle={unread > 0 ? `${unread} unread` : "You are up to date"}
        action={
          unread > 0 ? (
            <form action={markNotificationsRead}>
              <button type="submit" className="btn btn-ghost">
                Mark all read
              </button>
            </form>
          ) : null
        }
      />

      <div className="card">
        {notifications.length === 0 ? (
          <EmptyState title="Nothing yet" hint="Assignments, status moves and payments land here." />
        ) : (
          <ul className="divide-y divide-ink-100">
            {notifications.map((item) => {
              const row = (
                <div
                  className={`flex flex-wrap items-start justify-between gap-3 px-4 py-3 ${
                    item.readAt ? "" : "bg-brand-100/40"
                  }`}
                >
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-ink-900">{item.title}</p>
                    {item.body ? <p className="text-sm text-ink-500">{item.body}</p> : null}
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Badge>{humanise(item.type)}</Badge>
                    <span className="text-xs text-ink-500">{formatDateTime(item.createdAt)}</span>
                  </div>
                </div>
              );

              return (
                <li key={item.id}>
                  {item.link ? (
                    <Link href={item.link} className="block hover:bg-ink-50">
                      {row}
                    </Link>
                  ) : (
                    row
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </>
  );
}
