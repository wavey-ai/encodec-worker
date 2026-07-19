const baseUrl = process.env.ENCODEC_TEST_URL || "http://127.0.0.1:8787";
const originToken = process.env.ENCODEC_ORIGIN_TOKEN ?? "test-origin-token";
const authorizationHeaders = originToken ? { Authorization: `Bearer ${originToken}` } : {};
const profile = process.env.ENCODEC_TEST_PROFILE || "encodec_48khz_12kbps_1333ms";
const prefix = `queue-test-${Date.now()}`;

async function request(pathname, { owner, expect = 200, ...init } = {}) {
  const response = await fetch(`${baseUrl}${pathname}`, {
    ...init,
    headers: {
      ...authorizationHeaders,
      "X-Encodec-Owner": owner,
      ...(init.headers || {}),
    },
  });
  const body = await response.json();
  if (response.status !== expect) {
    throw new Error(`Expected HTTP ${expect}, got ${response.status}: ${body.error || "request_failed"} ${body.message || ""}`);
  }
  return body;
}

async function create(owner, admission, expect) {
  return request("/v1/sessions", {
    owner,
    expect,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile, segmentCount: 1, admission }),
  });
}

async function get(session, owner) {
  const body = await request(`/v1/sessions/${session.id}`, {
    owner,
    headers: { "X-Encodec-Session-Token": session.sessionToken },
  });
  return { ...body, session: { ...body.session, sessionToken: session.sessionToken } };
}

async function cancel(session, owner) {
  return request(`/v1/sessions/${session.id}`, {
    owner,
    method: "DELETE",
    headers: { "X-Encodec-Session-Token": session.sessionToken },
  });
}

const ownerA = `${prefix}-active`;
const ownerB = `${prefix}-queued`;

const active = (await create(ownerA, "try", 201)).session;
if (active.state !== "active") throw new Error(`Expected first session active, got ${active.state}`);

const busy = await create(`${prefix}-busy`, "try", 409);
if (busy.error !== "remote_busy") throw new Error(`Expected remote_busy, got ${busy.error}`);

const queued = (await create(ownerB, "queue", 202)).session;
if (queued.state !== "queued") throw new Error(`Expected queued session, got ${queued.state}`);
if (queued.queuePosition !== 1) throw new Error(`Expected queue position 1, got ${queued.queuePosition}`);

const duplicate = (await create(ownerB, "queue", 202)).session;
if (duplicate.state !== "queued") throw new Error(`Expected duplicate owner queue request to be accepted, got ${duplicate.state}`);

await cancel(active, ownerA);

const promoted = (await get(queued, ownerB)).session;
if (promoted.state !== "active") throw new Error(`Expected queued session to become active, got ${promoted.state}`);

await cancel(promoted, ownerB);
await cancel(duplicate, ownerB);

console.log(JSON.stringify({
  ok: true,
  profile,
  busyError: busy.error,
  duplicateState: duplicate.state,
  promotedState: promoted.state,
}, null, 2));
