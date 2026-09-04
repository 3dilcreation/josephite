import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE } from "@/lib/session-cookie";

/**
 * Cheap gate only — it checks that a session cookie is present so signed-out
 * visitors bounce to /login without a database round trip. Real authorisation
 * happens per page via requireUser()/scopeWhere(), which verifies the token and
 * re-reads the user's current role.
 */
export function middleware(request: NextRequest) {
  const hasSession = Boolean(request.cookies.get(SESSION_COOKIE)?.value);
  const { pathname } = request.nextUrl;

  if (!hasSession) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/leads/:path*",
    "/orders/:path*",
    "/customers/:path*",
    "/payments/:path*",
    "/notifications/:path*",
    "/admin/:path*",
  ],
};
