/**
 * Kept separate from session.ts so the Edge middleware can read the cookie name
 * without pulling the JWT library (and its Node-only APIs) into the Edge bundle.
 */
export const SESSION_COOKIE = "3dil_session";
