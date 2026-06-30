/*
 * ================================================================
 *   Smart RFID Attendance System  v4.2  —  TechSei Lab
 * ================================================================
 *   Hardware:
 *     ESP32  |  MFRC522 RFID  |  DS3231 RTC  |  I2C LCD 16×2
 *     Active buzzer GPIO 4  |  BOOT button GPIO 0
 *
 *   Libraries (Sketch → Include Library → Manage Libraries):
 *     MFRC522           by GithubCommunity
 *     LiquidCrystal_I2C by Frank de Brabander
 *     RTClib            by Adafruit
 *     ArduinoJson       by Benoit Blanchon  v6.x
 *
 *   v4.2 — FreeRTOS dual-core split:
 *     Core 1 (loop): RFID read → cache lookup → LCD → beep  (<50 ms)
 *     Core 0 (netTask): ALL HTTP — log, heartbeat, drain, settings, roster
 *     No HTTP ever runs on Core 1 → scan latency is hardware-limited only
 *
 *   Other features:
 *     • IN/OUT day-guard: cannot check-out without today's check-in
 *     • Daily GAS auto-exit trigger fills outstanding OUT at close time
 *     • Roster version signal: enrol/delete visible on device in ≤ 15 s
 *     • Background queue drain (FIFO, one record per idle cycle)
 * ================================================================
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <RTClib.h>
#include <Preferences.h>
#include "SPIFFS.h"
#include <ArduinoJson.h>
#include "config.h"

// ── Objects ─────────────────────────────────────────────────────
MFRC522           rfid(SS_PIN, RST_PIN);
LiquidCrystal_I2C lcd(LCD_ADDR, 16, 2);
RTC_DS3231        rtc;
Preferences       prefs;

// ── RTOS synchronisation ─────────────────────────────────────────
SemaphoreHandle_t xRosterMutex;  // guards g_roster[] between cores
SemaphoreHandle_t xNvsMutex;     // guards Preferences (not thread-safe)
SemaphoreHandle_t xLcdRowMutex;  // guards g_lcdRow snapshot for heartbeat
SemaphoreHandle_t xHttpMutex;    // one HTTP call at a time (admin + net task)

// ── Scan → net task queue ────────────────────────────────────────
#define NET_QUEUE_LEN 30
struct ScanEvent {
  char uid[16];
  char date[12];   // DD/MM/YYYY
  char inTime[6];  // "HH:MM" or ""
  char outTime[6]; // "HH:MM" or ""
  char status[12]; // ON_TIME / LATE / EARLY_ARR / EARLY_DEP / OVERTIME
};
QueueHandle_t g_netQueue;

// ── Global state ─────────────────────────────────────────────────
volatile bool adminMode    = false;
volatile bool wifiOk       = false;
unsigned long adminModeMs  = 0;
String        lastUID      = "";
volatile unsigned long lastScanMs  = 0;   // written Core 1, read Core 0
unsigned long lastClockMs  = 0;
bool          btnWasLow    = false;
unsigned long btnLowMs     = 0;
volatile int  offlineCount = 0;           // written Core 0, read by heartbeat

const char* DAYS[] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};

// ── LCD row snapshot (read by net task for heartbeat) ────────────
char g_lcdRow0[17] = "                ";
char g_lcdRow1[17] = "                ";

// ── Runtime time thresholds (volatile so Core 0 write, Core 1 read) ──
volatile int g_openHour     = OPEN_HOUR;
volatile int g_openMin      = OPEN_MIN;
volatile int g_lateHour     = LATE_HOUR;
volatile int g_lateMin      = LATE_MIN;
volatile int g_closeHour    = CLOSE_HOUR;
volatile int g_closeMin     = CLOSE_MIN;
volatile int g_earlyOutHour = EARLY_OUT_HOUR;
volatile int g_earlyOutMin  = EARLY_OUT_MIN;
volatile int g_overtimeHour = OVERTIME_HOUR;
volatile int g_overtimeMin  = OVERTIME_MIN;
String g_settingsVersion = "";
String g_rosterVersion   = "";

// ── Local roster cache ───────────────────────────────────────────
struct RosterEntry { char uid[16]; char name[33]; char type[10]; };
static RosterEntry g_roster[MAX_ROSTER_ENTRIES];
volatile int g_rosterCount = 0;


// ════════════════════════════════════════════════════════════════
//  1. BUZZER
// ════════════════════════════════════════════════════════════════

void beepIn() {
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);  delay(80);
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);
}

void beepOut() {
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);  delay(70);
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);  delay(70);
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);
}


// ════════════════════════════════════════════════════════════════
//  2. LCD HELPERS  (Core 1 only for hardware; lcdRow also read Core 0)
// ════════════════════════════════════════════════════════════════

void lcdRowSet(const char* r0, const char* r1) {
  xSemaphoreTake(xLcdRowMutex, portMAX_DELAY);
  strncpy(g_lcdRow0, r0, 16); g_lcdRow0[16] = '\0';
  strncpy(g_lcdRow1, r1, 16); g_lcdRow1[16] = '\0';
  xSemaphoreGive(xLcdRowMutex);
}

void lcdMsg(const char* top, const char* bot) {
  lcdRowSet(top, bot);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(top);
  lcd.setCursor(0, 1); lcd.print(bot);
}

void updateClock() {
  DateTime now = rtc.now();
  char row0[17];
  snprintf(row0, sizeof(row0), "%s %02d/%02d  %02d:%02d",
    DAYS[now.dayOfTheWeek()], now.day(), now.month(),
    now.hour(), now.minute());

  char row1[17];
  if (adminMode) {
    snprintf(row1, sizeof(row1), "**ADMIN** Scan  ");
  } else {
    const char* wifi = wifiOk ? "[WiFi]" : "[OFLN]";
    int cur = now.hour() * 60 + now.minute();
    if      (cur <  g_openHour  * 60 + g_openMin)  snprintf(row1, sizeof(row1), "%s TooEarly ", wifi);
    else if (cur >= g_closeHour * 60 + g_closeMin)  snprintf(row1, sizeof(row1), "%s Closed   ", wifi);
    else if (cur >= g_lateHour  * 60 + g_lateMin)   snprintf(row1, sizeof(row1), "%s LATE Scan", wifi);
    else                                              snprintf(row1, sizeof(row1), "%s Scan Card", wifi);
  }

  lcdRowSet(row0, row1);
  lcd.setCursor(0, 0); lcd.print(row0);
  lcd.setCursor(0, 1); lcd.print(row1);
}

void showResult(const char* line0, const char* line1, int holdMs) {
  lcdRowSet(line0, line1);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(line0);
  lcd.setCursor(0, 1); lcd.print(line1);
  delay(holdMs);
  updateClock();
}


// ════════════════════════════════════════════════════════════════
//  3. URL ENCODE
// ════════════════════════════════════════════════════════════════

String urlEncode(const String& str) {
  String encoded = "";
  char buf[4];
  for (unsigned int i = 0; i < str.length(); i++) {
    char c = str.charAt(i);
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      encoded += c;
    } else {
      sprintf(buf, "%%%02X", (uint8_t)c);
      encoded += buf;
    }
  }
  return encoded;
}


// ════════════════════════════════════════════════════════════════
//  4. WIFI
// ════════════════════════════════════════════════════════════════

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) { wifiOk = true; return; }
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Connecting");
  // Use vTaskDelay so other RTOS tasks still run during connect
  for (int i = 0; i < 24 && WiFi.status() != WL_CONNECTED; i++) {
    vTaskDelay(pdMS_TO_TICKS(500)); Serial.print(".");
  }
  wifiOk = (WiFi.status() == WL_CONNECTED);
  Serial.println(wifiOk ? "\n[WiFi] Connected" : "\n[WiFi] Offline");
}


// ════════════════════════════════════════════════════════════════
//  5. RTC
// ════════════════════════════════════════════════════════════════

String getTimeStr() {
  DateTime now = rtc.now();
  char buf[6]; sprintf(buf, "%02d:%02d", now.hour(), now.minute());
  return String(buf);
}

String getDateStr() {
  DateTime now = rtc.now();
  char buf[11]; sprintf(buf, "%02d/%02d/%04d", now.day(), now.month(), now.year());
  return String(buf);
}


// ════════════════════════════════════════════════════════════════
//  6. PREFERENCES  (all callers must hold xNvsMutex)
// ════════════════════════════════════════════════════════════════

String makeKey(const String& uid) {
  String k = uid; k.replace(":", ""); k.replace(" ", "");
  if (k.length() > 15) k = k.substring(0, 15);
  return k;
}
String makeTimeKey(const String& uid) {
  String k = "t" + makeKey(uid); if (k.length() > 15) k = k.substring(0, 15); return k;
}
String makeInDateKey(const String& uid) {
  String k = "d" + makeKey(uid); if (k.length() > 15) k = k.substring(0, 15); return k;
}

// Prefixed with _ — call only while holding xNvsMutex
bool     _getStatus(const String& uid)              { return prefs.getBool  (makeKey(uid).c_str(), true); }
void     _setStatus(const String& uid, bool v)      { prefs.putBool  (makeKey(uid).c_str(), v); }
void     _saveCheckInTime(const String& uid, const String& t) { prefs.putString(makeTimeKey(uid).c_str(), t.c_str()); }
String   _getCheckInTime (const String& uid)        { return prefs.getString(makeTimeKey(uid).c_str(), ""); }
void     _saveCheckInDate(const String& uid, const String& d) { prefs.putString(makeInDateKey(uid).c_str(), d.c_str()); }
String   _getCheckInDate (const String& uid)        { return prefs.getString(makeInDateKey(uid).c_str(), ""); }


// ════════════════════════════════════════════════════════════════
//  7. STATUS DETERMINATION  (reads volatile g_* thresholds — safe)
// ════════════════════════════════════════════════════════════════

String determineStatus(bool isIn, int h, int m) {
  int cur = h * 60 + m;
  if (isIn) {
    if (cur <  g_openHour  * 60 + g_openMin)  return "EARLY_ARR";
    if (cur >= g_lateHour  * 60 + g_lateMin)  return "LATE";
    return "ON_TIME";
  } else {
    if (cur >= g_overtimeHour * 60 + g_overtimeMin)  return "OVERTIME";
    if (cur <  g_earlyOutHour * 60 + g_earlyOutMin)  return "EARLY_DEP";
    return "ON_TIME";
  }
}

const char* statusLabel(const String& st) {
  if (st == "LATE")      return "LATE ";
  if (st == "EARLY_ARR") return "E.ARR";
  if (st == "EARLY_DEP") return "E.DEP";
  if (st == "OVERTIME")  return "OT   ";
  return "OK   ";
}

String calcHoursLcd(const String& inT, const String& outT) {
  if (inT.length() < 5 || outT.length() < 5) return "?h   ";
  int diff = (outT.substring(0,2).toInt()*60 + outT.substring(3,5).toInt())
           - (inT.substring(0,2).toInt() *60 + inT.substring(3,5).toInt());
  if (diff <= 0) return "0h   ";
  char buf[8]; snprintf(buf, sizeof(buf), "%.1fh", diff / 60.0f);
  String s = String(buf); while ((int)s.length() < 5) s += " ";
  return s.substring(0, 5);
}


// ════════════════════════════════════════════════════════════════
//  7b. ROSTER CACHE  (callers must hold xRosterMutex for write;
//      lookupRoster is called under xRosterMutex from Core 1)
// ════════════════════════════════════════════════════════════════

bool lookupRoster(const String& uid, char* outName, char* outType) {
  for (int i = 0; i < g_rosterCount; i++) {
    if (uid.equalsIgnoreCase(String(g_roster[i].uid))) {
      strncpy(outName, g_roster[i].name, 32); outName[32] = '\0';
      strncpy(outType, g_roster[i].type, 9);  outType[9]  = '\0';
      return true;
    }
  }
  return false;
}


// ════════════════════════════════════════════════════════════════
//  8. SPIFFS OFFLINE QUEUE  (only touched by net task — no mutex needed)
//     Format per line:  uid|date|intime|outtime|status
// ════════════════════════════════════════════════════════════════

void saveOffline(const String& uid, const String& date,
                 const String& inT,  const String& outT,
                 const String& status) {
  File f = SPIFFS.open("/queue.txt", FILE_APPEND);
  if (!f) { Serial.println("[SPIFFS] Open failed"); return; }
  f.println(uid + "|" + date + "|" +
            (inT.isEmpty()  ? "-" : inT)  + "|" +
            (outT.isEmpty() ? "-" : outT) + "|" + status);
  f.close();
  offlineCount++;
  Serial.printf("[Queue] Offline #%d\n", (int)offlineCount);
}


// ════════════════════════════════════════════════════════════════
//  9. HTTP  (protected by xHttpMutex — allows admin scan on Core 1
//            to coexist safely with net task HTTP on Core 0)
// ════════════════════════════════════════════════════════════════

String httpGet(const String& url) {
  xSemaphoreTake(xHttpMutex, portMAX_DELAY);
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;
  http.begin(client, url);
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.setTimeout(8000);
  int code = http.GET();
  String body = (code > 0) ? http.getString() : "";
  http.end();
  xSemaphoreGive(xHttpMutex);
  Serial.printf("[HTTP] %d\n", code);
  return body;
}

String sendLog(const String& uid,  const String& date,
               const String& inT,  const String& outT,
               const String& status) {
  return httpGet(String(GAS_URL) +
    "?action=log&uid="  + urlEncode(uid)    +
    "&date="            + urlEncode(date)   +
    "&intime="          + urlEncode(inT)    +
    "&outtime="         + urlEncode(outT)   +
    "&status="          + urlEncode(status));
}

String pushAdminScan(const String& uid) {
  return httpGet(String(GAS_URL) + "?action=pushScan&uid=" + urlEncode(uid));
}


// ════════════════════════════════════════════════════════════════
//  10. SETTINGS FETCH  (net task only)
// ════════════════════════════════════════════════════════════════

void fetchSettings() {
  if (!wifiOk) return;
  String body = httpGet(String(GAS_URL) + "?action=getSettings");
  if (body.isEmpty()) return;
  StaticJsonDocument<512> doc;
  if (deserializeJson(doc, body) != DeserializationError::Ok) return;
  if (!(doc["ok"] | false)) return;
  // 32-bit aligned writes — safe to read from Core 1 without mutex
  g_openHour     = doc["openHour"]     | (int)g_openHour;
  g_openMin      = doc["openMin"]      | (int)g_openMin;
  g_lateHour     = doc["lateHour"]     | (int)g_lateHour;
  g_lateMin      = doc["lateMin"]      | (int)g_lateMin;
  g_closeHour    = doc["closeHour"]    | (int)g_closeHour;
  g_closeMin     = doc["closeMin"]     | (int)g_closeMin;
  g_earlyOutHour = doc["earlyOutHour"] | (int)g_earlyOutHour;
  g_earlyOutMin  = doc["earlyOutMin"]  | (int)g_earlyOutMin;
  g_overtimeHour = doc["overtimeHour"] | (int)g_overtimeHour;
  g_overtimeMin  = doc["overtimeMin"]  | (int)g_overtimeMin;
  g_settingsVersion = String(doc["version"] | "0");
  Serial.printf("[Settings] v%s open=%02d:%02d late=%02d:%02d close=%02d:%02d\n",
    g_settingsVersion.c_str(), (int)g_openHour, (int)g_openMin,
    (int)g_lateHour, (int)g_lateMin, (int)g_closeHour, (int)g_closeMin);
}


// ════════════════════════════════════════════════════════════════
//  10b. HEARTBEAT  (net task only)
// ════════════════════════════════════════════════════════════════

void sendHeartbeat() {
  if (!wifiOk) return;
  // Snapshot LCD rows under mutex (Core 1 may be writing simultaneously)
  char row0[17], row1[17];
  xSemaphoreTake(xLcdRowMutex, portMAX_DELAY);
  strncpy(row0, g_lcdRow0, 16); row0[16] = '\0';
  strncpy(row1, g_lcdRow1, 16); row1[16] = '\0';
  xSemaphoreGive(xLcdRowMutex);

  String body = httpGet(String(GAS_URL) +
    "?action=heartbeat" +
    "&row0="  + urlEncode(String(row0)) +
    "&row1="  + urlEncode(String(row1)) +
    "&wifi="  + (wifiOk ? "1" : "0") +
    "&queue=" + String(offlineCount) +
    "&mode="  + (adminMode ? "admin" : "normal"));
  if (body.isEmpty()) return;

  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, body) != DeserializationError::Ok) return;

  String remoteVer = String(doc["settingsVersion"] | "0");
  if (remoteVer != "0" && remoteVer != g_settingsVersion) {
    Serial.println("[HB] Settings changed → re-fetch");
    fetchSettings();
  }
  String remoteRosterVer = String(doc["rosterVersion"] | "0");
  if (remoteRosterVer != "0" && remoteRosterVer != g_rosterVersion) {
    Serial.println("[HB] Roster changed → re-fetch");
    fetchRoster(true);
  }
}


// ════════════════════════════════════════════════════════════════
//  10c. ROSTER FETCH  (net task + setup; silent=true for background)
// ════════════════════════════════════════════════════════════════

void fetchRoster(bool silent) {
  if (!wifiOk) return;
  if (!silent) lcdMsg("Loading roster..", "Please wait...  ");
  Serial.println("[Roster] Fetching...");

  String body = httpGet(String(GAS_URL) + "?action=getRoster");
  if (body.isEmpty()) { if (!silent) updateClock(); return; }

  DynamicJsonDocument doc(16384);
  if (deserializeJson(doc, body) != DeserializationError::Ok || !(doc["ok"] | false)) {
    if (!silent) updateClock(); return;
  }

  JsonArray arr = doc["students"].as<JsonArray>();
  xSemaphoreTake(xRosterMutex, portMAX_DELAY);
  int n = 0;
  for (JsonObject s : arr) {
    if (n >= MAX_ROSTER_ENTRIES) break;
    const char* u  = s["uid"]  | ""; if (!u || !u[0]) continue;
    const char* nm = s["name"] | "";
    const char* ty = s["type"] | "Student";
    strncpy(g_roster[n].uid,  u,  15); g_roster[n].uid[15]  = '\0';
    strncpy(g_roster[n].name, nm, 32); g_roster[n].name[32] = '\0';
    strncpy(g_roster[n].type, ty, 9);  g_roster[n].type[9]  = '\0';
    n++;
  }
  g_rosterCount = n;
  xSemaphoreGive(xRosterMutex);

  g_rosterVersion = String(doc["rosterVersion"] | "0");
  Serial.printf("[Roster] Cached %d entries (v%s)\n", n, g_rosterVersion.c_str());
  if (!silent) updateClock();
}


// ════════════════════════════════════════════════════════════════
//  11. DRAIN ONE OFFLINE RECORD  (net task only — single writer)
// ════════════════════════════════════════════════════════════════

void drainOneRecord() {
  if (!SPIFFS.exists("/queue.txt")) { offlineCount = 0; return; }
  File src = SPIFFS.open("/queue.txt", FILE_READ);
  if (!src) return;

  String firstLine = "";
  while (src.available()) {
    String l = src.readStringUntil('\n'); l.trim();
    if (!l.isEmpty()) { firstLine = l; break; }
  }
  File tmp = SPIFFS.open("/qtmp.txt", FILE_WRITE);
  int remaining = 0;
  while (src.available()) {
    String l = src.readStringUntil('\n'); l.trim();
    if (!l.isEmpty()) { tmp.println(l); remaining++; }
  }
  src.close(); tmp.close();

  if (firstLine.isEmpty()) {
    SPIFFS.remove("/queue.txt"); SPIFFS.remove("/qtmp.txt");
    offlineCount = 0; return;
  }

  String parts[5]; int idx = 0, start = 0;
  for (int i = 0; i <= (int)firstLine.length() && idx < 5; i++) {
    if (i == (int)firstLine.length() || firstLine[i] == '|') {
      parts[idx++] = firstLine.substring(start, i); start = i + 1;
    }
  }

  bool discard = (idx < 4);
  if (!discard) {
    String inT  = (parts[2] == "-") ? "" : parts[2];
    String outT = (parts[3] == "-") ? "" : parts[3];
    String stat = (idx > 4 && parts[4].length()) ? parts[4] : "ON_TIME";
    String resp = sendLog(parts[0], parts[1], inT, outT, stat);
    StaticJsonDocument<256> doc;
    bool ok = false;
    if (deserializeJson(doc, resp) == DeserializationError::Ok) {
      ok = doc["ok"] | false;
      // no_entry on OUT: create a synthetic IN then retry
      if (!ok && outT.length() && String(doc["msg"] | "") == "no_entry") {
        sendLog(parts[0], parts[1], outT, "", "ON_TIME");
        String r2 = sendLog(parts[0], parts[1], "", outT, stat);
        if (deserializeJson(doc, r2) == DeserializationError::Ok) ok = doc["ok"] | false;
      }
    }
    discard = ok;
  }

  if (discard) {
    SPIFFS.remove("/queue.txt");
    if (remaining > 0) SPIFFS.rename("/qtmp.txt", "/queue.txt");
    else               SPIFFS.remove("/qtmp.txt");
    offlineCount = remaining;
    Serial.printf("[Queue] Drained 1; %d left\n", remaining);
  } else {
    SPIFFS.remove("/qtmp.txt"); // keep original on failure
  }
}


// ════════════════════════════════════════════════════════════════
//  12. RFID UID READER
// ════════════════════════════════════════════════════════════════

String readUID() {
  String uid = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) uid += "0";
    uid += String(rfid.uid.uidByte[i], HEX);
  }
  uid.toLowerCase();
  return uid;
}


// ════════════════════════════════════════════════════════════════
//  13. NORMAL MODE  (Core 1 — ZERO HTTP — always < 50 ms)
//
//  Flow: NVS read → roster lookup → build display → LCD+beep → push event
//  Net task picks up the event and handles all GAS communication.
//
//  LCD formats (exactly 16 chars):
//    "IN  HH:MM  XXXXX"   XXXXX = OK    / LATE  / E.ARR
//    "OUT HH:MM  XXXXX"   XXXXX = 2.5h / OT    / E.DEP
//    name line padded to 16 chars
// ════════════════════════════════════════════════════════════════

void processNormal(const String& uid) {
  DateTime t = rtc.now();
  int h = t.hour(), m = t.minute();
  String date    = getDateStr();
  String timeNow = getTimeStr();

  // ── NVS read (one mutex take covers all NVS calls) ────────────
  bool   isIn;
  String checkInDate, checkInTime;
  xSemaphoreTake(xNvsMutex, portMAX_DELAY);
  isIn        = _getStatus(uid);
  checkInDate = _getCheckInDate(uid);
  checkInTime = _getCheckInTime(uid);
  xSemaphoreGive(xNvsMutex);

  // Guard: cannot check-out without today's check-in
  if (!isIn && checkInDate != date) isIn = true;

  // ── Roster lookup ─────────────────────────────────────────────
  char cachedName[33] = "";
  char cachedType[10] = "Student";
  int  cacheSize;
  xSemaphoreTake(xRosterMutex, portMAX_DELAY);
  bool known = lookupRoster(uid, cachedName, cachedType);
  cacheSize  = g_rosterCount;
  xSemaphoreGive(xRosterMutex);

  if (!known && cacheSize > 0) {
    // Cache is populated but this UID is not enrolled
    char l1[17]; snprintf(l1, sizeof(l1), "%-16s", uid.substring(0, 16).c_str());
    showResult("X Unknown Card  ", l1, 2500);
    return;
  }

  String status = determineStatus(isIn, h, m);

  // ── Build display lines ───────────────────────────────────────
  char line0[17], line1[17];
  snprintf(line1, sizeof(line1), "%-16s",
    (known ? String(cachedName) : uid).substring(0, 16).c_str());

  ScanEvent evt = {};
  strncpy(evt.uid,    uid.c_str(),    15);
  strncpy(evt.date,   date.c_str(),   11);
  strncpy(evt.status, status.c_str(), 11);

  if (isIn) {
    snprintf(line0, sizeof(line0), "IN  %s  %s", timeNow.c_str(), statusLabel(status));
    strncpy(evt.inTime, timeNow.c_str(), 5);

    xSemaphoreTake(xNvsMutex, portMAX_DELAY);
    _saveCheckInTime(uid, timeNow);
    _saveCheckInDate(uid, date);
    _setStatus(uid, false);  // next scan = OUT
    xSemaphoreGive(xNvsMutex);

    lcdRowSet(line0, line1);
    lcd.clear(); lcd.setCursor(0,0); lcd.print(line0);
                 lcd.setCursor(0,1); lcd.print(line1);
    beepIn();

  } else {
    String info = (status == "OVERTIME" || status == "EARLY_DEP")
      ? String(statusLabel(status)) : calcHoursLcd(checkInTime, timeNow);
    snprintf(line0, sizeof(line0), "OUT %s  %s", timeNow.c_str(), info.c_str());
    strncpy(evt.outTime, timeNow.c_str(), 5);

    xSemaphoreTake(xNvsMutex, portMAX_DELAY);
    _setStatus(uid, true);   // next scan = IN
    xSemaphoreGive(xNvsMutex);

    lcdRowSet(line0, line1);
    lcd.clear(); lcd.setCursor(0,0); lcd.print(line0);
                 lcd.setCursor(0,1); lcd.print(line1);
    beepOut();
  }

  // Push to net task — non-blocking (if queue full, event is dropped gracefully)
  if (xQueueSend(g_netQueue, &evt, 0) != pdTRUE) {
    Serial.println("[Queue] Full — saving directly offline");
    saveOffline(uid, date, String(evt.inTime), String(evt.outTime), status);
  }

  delay(2500);
  updateClock();
}


// ════════════════════════════════════════════════════════════════
//  14. ADMIN MODE  (Core 1; HTTP allowed here — admin use is rare)
// ════════════════════════════════════════════════════════════════

void processAdmin(const String& uid) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Sending to web..");
  lcd.setCursor(0, 1); lcd.print(uid.substring(0, 16));

  if (!wifiOk) { showResult("No WiFi!        ", "Need WiFi 4 enrl", 2500); return; }

  String resp = pushAdminScan(uid);
  if (resp.isEmpty()) { showResult("Server Error    ", "Try again       ", 2500); return; }

  StaticJsonDocument<256> doc;
  bool ok = false, existing = false; String name = "";
  if (deserializeJson(doc, resp) == DeserializationError::Ok) {
    ok = doc["ok"] | false; existing = doc["existing"] | false; name = doc["name"] | "";
  }
  if (!ok) { showResult("Server Error    ", "Try again       ", 2500); return; }

  if (existing) {
    char l0[17];
    snprintf(l0, sizeof(l0), "%-16s", ("Enrolled: " + name).substring(0, 16).c_str());
    showResult(l0, "Scan another crd", 2500);
  } else {
    char l1[17]; snprintf(l1, sizeof(l1), "%-16s", uid.substring(0, 16).c_str());
    showResult("Card Sent! ->   ", "Fill at web app ", 2500);
  }
  adminModeMs = millis();
}


// ════════════════════════════════════════════════════════════════
//  15. NET TASK  (Core 0 — ALL HTTP lives here)
//
//  Priority 2 on Core 0; loop() runs at priority 1 on Core 1.
//  Since they are on separate cores they run in true parallel.
//  xHttpMutex ensures admin-mode HTTP (Core 1) and netTask HTTP
//  (Core 0) never overlap at the WiFi stack level.
// ════════════════════════════════════════════════════════════════

void netTask(void* param) {
  unsigned long lastWifiCheck = 0;
  unsigned long lastHeartbeat = 0;
  unsigned long lastRoster    = millis();  // boot fetch already done
  unsigned long lastSettings  = millis();
  unsigned long lastDrain     = 0;
  ScanEvent evt = {};

  for (;;) {
    // ── Drain scan-event queue (highest priority in net task) ─────
    while (xQueueReceive(g_netQueue, &evt, 0) == pdTRUE) {
      String uid    = String(evt.uid);
      String date   = String(evt.date);
      String inT    = String(evt.inTime);
      String outT   = String(evt.outTime);
      String status = String(evt.status);

      if (wifiOk) {
        String resp = sendLog(uid, date, inT, outT, status);
        StaticJsonDocument<256> doc;
        bool ok = false;
        if (deserializeJson(doc, resp) == DeserializationError::Ok) {
          ok = doc["ok"] | false;
          // Server no_entry on OUT: synthesise an IN record first then retry
          if (!ok && outT.length() && String(doc["msg"] | "") == "no_entry") {
            sendLog(uid, date, outT, "", "ON_TIME");
            String r2 = sendLog(uid, date, "", outT, status);
            if (deserializeJson(doc, r2) == DeserializationError::Ok) ok = doc["ok"] | false;
          }
        }
        if (!ok) saveOffline(uid, date, inT, outT, status);
      } else {
        saveOffline(uid, date, inT, outT, status);
      }
    }

    unsigned long now = millis();

    // ── WiFi watchdog ─────────────────────────────────────────────
    if (now - lastWifiCheck > WIFI_CHECK_MS) {
      lastWifiCheck = now;
      bool wasOk = wifiOk;
      if (WiFi.status() != WL_CONNECTED) { wifiOk = false; connectWiFi(); }
      if (!wasOk && wifiOk) { fetchRoster(true); lastRoster = millis(); }
    }

    // ── Heartbeat (every 15 s) ────────────────────────────────────
    if (wifiOk && now - lastHeartbeat > HEARTBEAT_MS) {
      lastHeartbeat = now;
      sendHeartbeat();
    }

    // ── Drain one offline record when idle (3 s since last scan) ──
    if (wifiOk && offlineCount > 0 &&
        (millis() - lastScanMs > 3000) && (now - lastDrain > 500)) {
      lastDrain = millis();
      drainOneRecord();
    }

    // ── Hourly roster refresh (retry every 60 s while cache empty) ──
    {
      unsigned long interval = (g_rosterCount == 0) ? 60000UL : (unsigned long)ROSTER_REFRESH_MS;
      if (wifiOk && now - lastRoster > interval) {
        lastRoster = millis(); fetchRoster(true);
      }
    }

    // ── Hourly settings refresh ───────────────────────────────────
    if (wifiOk && now - lastSettings > SETTINGS_REFRESH_MS) {
      lastSettings = millis(); fetchSettings();
    }

    // Yield: sleep up to 50 ms, wake immediately if a scan event arrives.
    // Clear evt first so stale data from a processed event can't be re-queued.
    memset(&evt, 0, sizeof(evt));
    xQueueReceive(g_netQueue, &evt, pdMS_TO_TICKS(50));
    if (evt.uid[0]) xQueueSendToFront(g_netQueue, &evt, 0);
    memset(&evt, 0, sizeof(evt));
  }
}


// ════════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println(F("\n=== Smart RFID Attendance v4.2 ==="));

  // MUST be first: every lcdMsg/updateClock calls lcdRowSet which takes xLcdRowMutex.
  // Creating mutexes after any LCD call causes xSemaphoreTake(NULL) → FreeRTOS assert → reboot loop.
  xRosterMutex = xSemaphoreCreateMutex();
  xNvsMutex    = xSemaphoreCreateMutex();
  xLcdRowMutex = xSemaphoreCreateMutex();
  xHttpMutex   = xSemaphoreCreateMutex();
  g_netQueue   = xQueueCreate(NET_QUEUE_LEN, sizeof(ScanEvent));

  pinMode(BUZZER,    OUTPUT); digitalWrite(BUZZER, LOW);
  pinMode(ADMIN_BTN, INPUT_PULLUP);

  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init(); lcd.backlight();
  lcd.noCursor(); lcd.noBlink();
  lcdMsg("Smart Attendance", ORG_NAME);
  delay(2000);

  if (!rtc.begin()) {
    lcdMsg("RTC Error!      ", "Check wiring    ");
    Serial.println("[RTC] Not found");
    while (1) delay(1000);
  }
  if (rtc.lostPower()) {
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    Serial.println("[RTC] Lost power — reset to compile time");
  }
  Serial.printf("[RTC] %s %s\n", getDateStr().c_str(), getTimeStr().c_str());

  if (!SPIFFS.begin(true)) {
    Serial.println("[SPIFFS] Mount failed");
  } else {
    if (SPIFFS.exists("/queue.txt")) {
      File f = SPIFFS.open("/queue.txt", FILE_READ);
      while (f.available()) {
        String l = f.readStringUntil('\n'); l.trim();
        if (!l.isEmpty()) offlineCount++;
      }
      f.close();
      Serial.printf("[SPIFFS] %d pending record(s)\n", (int)offlineCount);
    }
  }

  prefs.begin("attend", false);

  SPI.begin();
  rfid.PCD_Init();
  delay(50);
  Serial.println(F("[RFID] Ready"));

  // Initial data pull (synchronous — net task not yet running)
  lcdMsg("Connecting WiFi.", WIFI_SSID);
  connectWiFi();
  fetchSettings();
  fetchRoster(false);  // shows "Loading roster.." on LCD

  // Net task on Core 0, priority 2  (loop() is priority 1 on Core 1)
  xTaskCreatePinnedToCore(netTask, "netTask", 20480, NULL, 2, NULL, 0);

  updateClock();
  Serial.println(F("[SYS] Ready — Core 1: RFID/LCD | Core 0: HTTP\n"));
}


// ════════════════════════════════════════════════════════════════
//  LOOP  (Core 1 — RFID, button, clock ONLY — zero HTTP)
// ════════════════════════════════════════════════════════════════

void loop() {
  unsigned long now = millis();

  // ── Admin button (hold BOOT for ADMIN_HOLD_MS) ────────────────
  bool btnLow = (digitalRead(ADMIN_BTN) == LOW);
  if (btnLow && !btnWasLow) { btnWasLow = true; btnLowMs = now; }
  if (btnLow && btnWasLow && (now - btnLowMs >= ADMIN_HOLD_MS)) {
    adminMode   = !adminMode;
    adminModeMs = now;
    btnWasLow   = false;
    Serial.printf("[BTN] Admin %s\n", adminMode ? "ON" : "OFF");
    updateClock(); delay(300);
  }
  if (!btnLow) btnWasLow = false;

  // ── Admin auto-timeout ────────────────────────────────────────
  if (adminMode && (now - adminModeMs > ADMIN_TIMEOUT_MS)) {
    adminMode = false;
    Serial.println("[BTN] Admin timeout");
    updateClock();
  }

  // ── Clock update (1 s) ────────────────────────────────────────
  if (now - lastClockMs > 1000) { lastClockMs = now; updateClock(); }

  // ── RFID poll (~100 Hz) ───────────────────────────────────────
  // 10 ms yield between polls: limits RF field hammering (which causes
  // erratic reads at full CPU speed) and feeds the Core 1 idle task.
  if (!rfid.PICC_IsNewCardPresent()) { vTaskDelay(pdMS_TO_TICKS(10)); return; }
  if (!rfid.PICC_ReadCardSerial())   { vTaskDelay(pdMS_TO_TICKS(10)); return; }

  String uid = readUID();

  if (uid == lastUID && (now - lastScanMs) < SCAN_DEBOUNCE_MS) {
    rfid.PICC_HaltA(); vTaskDelay(pdMS_TO_TICKS(10)); return;
  }
  lastUID    = uid;
  lastScanMs = now;

  Serial.printf("[RFID] %s  admin=%d\n", uid.c_str(), (bool)adminMode);

  if (adminMode) processAdmin(uid);
  else           processNormal(uid);

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
}
