import { requireUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { Sidebar } from "@/components/sidebar";
import { Topbar } from "@/components/topbar";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const user = await requireUser();
  const unread = await db.notification.count({ where: { userId: user.id, readAt: null } });

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <Sidebar role={user.role} unread={unread} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar user={user} unread={unread} />
        <main className="min-w-0 flex-1 px-4 py-6 sm:px-6">{children}</main>
      </div>
    </div>
  );
}
