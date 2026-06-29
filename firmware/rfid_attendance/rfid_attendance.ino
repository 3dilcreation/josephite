/*
 * ================================================================
 *   Smart RFID Attendance System  v4.1  —  TechSei Lab
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
 *   Features (v4.1):
 *     • Check-in / check-out toggle per UID (Preferences NVS, survives reboot)
 *     • DS3231 RTC — accurate time without internet
 *     • SPIFFS offline queue — auto-uploads when WiFi returns
 *     • Buzzer ONLY on check-in / check-out (no boot beeps, no error beeps)
 *     • Admin mode (hold BOOT 1 s) — scan → enroll via web dashboard
 *     • Student / Staff / Others support
 *     • Time-based status: ON_TIME / LATE / EARLY_ARR / EARLY_DEP / OVERTIME
 *     • All time thresholds configurable in config.h — no recompile needed
 *     • LCD shows hours worked on check-out (e.g. "OUT 17:30  8.0h ")
 *     • Idle screen shows contextual hint (LATE, Closed, etc.)
 * ================================================================
 */

#include <WiFi.h>
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

// ── Global state ─────────────────────────────────────────────────
bool          wifiOk       = false;
bool          adminMode    = false;
unsigned long adminModeMs  = 0;
String        lastUID      = "";
unsigned long lastScanMs   = 0;
unsigned long lastWifiMs   = 0;
unsigned long lastClockMs  = 0;
bool          btnWasLow    = false;
unsigned long btnLowMs     = 0;
int           offlineCount = 0;

const char* DAYS[] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};


// ════════════════════════════════════════════════════════════════
//  1. BUZZER — ONLY check-in and check-out confirmations
// ════════════════════════════════════════════════════════════════

void beepIn() {
  // Two short beeps = welcome / check-in
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);  delay(80);
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);
}

void beepOut() {
  // Three short beeps = goodbye / check-out
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);  delay(70);
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);  delay(70);
  digitalWrite(BUZZER, HIGH); delay(120);
  digitalWrite(BUZZER, LOW);
}


// ════════════════════════════════════════════════════════════════
//  2. LCD HELPERS
// ════════════════════════════════════════════════════════════════

void lcdMsg(const char* top, const char* bot) {
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
    if (cur < OPEN_HOUR * 60 + OPEN_MIN) {
      snprintf(row1, sizeof(row1), "%s TooEarly ", wifi);
    } else if (cur >= CLOSE_HOUR * 60 + CLOSE_MIN) {
      snprintf(row1, sizeof(row1), "%s Closed   ", wifi);
    } else if (cur >= LATE_HOUR * 60 + LATE_MIN) {
      snprintf(row1, sizeof(row1), "%s LATE Scan", wifi);
    } else {
      snprintf(row1, sizeof(row1), "%s Scan Card", wifi);
    }
  }

  lcd.setCursor(0, 0); lcd.print(row0);
  lcd.setCursor(0, 1); lcd.print(row1);
}

void showResult(const char* line0, const char* line1, int holdMs) {
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
  for (int i = 0; i < 24 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500); Serial.print(".");
  }
  wifiOk = (WiFi.status() == WL_CONNECTED);
  Serial.println(wifiOk ? "\n[WiFi] Connected" : "\n[WiFi] Offline mode");
}


// ════════════════════════════════════════════════════════════════
//  5. RTC — formatted strings
// ════════════════════════════════════════════════════════════════

String getTimeStr() {
  DateTime now = rtc.now();
  char buf[6];
  sprintf(buf, "%02d:%02d", now.hour(), now.minute());
  return String(buf);
}

String getDateStr() {
  DateTime now = rtc.now();
  char buf[11];
  sprintf(buf, "%02d/%02d/%04d", now.day(), now.month(), now.year());
  return String(buf);
}


// ════════════════════════════════════════════════════════════════
//  6. PREFERENCES — per-card IN/OUT status and check-in time
// ════════════════════════════════════════════════════════════════

// Strip colons/spaces, cap at 15 chars (Preferences key limit)
String makeKey(const String& uid) {
  String k = uid;
  k.replace(":", "");
  k.replace(" ", "");
  if (k.length() > 15) k = k.substring(0, 15);
  return k;
}

// "t" prefix + first 14 chars of uid key = max 15 chars
String makeTimeKey(const String& uid) {
  String k = "t" + makeKey(uid);
  if (k.length() > 15) k = k.substring(0, 15);
  return k;
}

// Returns true = next scan is IN (default for new cards)
bool getStatus(const String& uid) {
  return prefs.getBool(makeKey(uid).c_str(), true);
}

void setStatus(const String& uid, bool nextIsIn) {
  prefs.putBool(makeKey(uid).c_str(), nextIsIn);
}

void saveCheckInTime(const String& uid, const String& t) {
  prefs.putString(makeTimeKey(uid).c_str(), t.c_str());
}

String getCheckInTime(const String& uid) {
  return prefs.getString(makeTimeKey(uid).c_str(), "");
}


// ════════════════════════════════════════════════════════════════
//  7. STATUS DETERMINATION (uses config.h thresholds)
// ════════════════════════════════════════════════════════════════

String determineStatus(bool isIn, int h, int m) {
  int cur = h * 60 + m;
  if (isIn) {
    if (cur <  OPEN_HOUR * 60 + OPEN_MIN)  return "EARLY_ARR";
    if (cur >= LATE_HOUR * 60 + LATE_MIN)  return "LATE";
    return "ON_TIME";
  } else {
    if (cur >= OVERTIME_HOUR  * 60 + OVERTIME_MIN)   return "OVERTIME";
    if (cur <  EARLY_OUT_HOUR * 60 + EARLY_OUT_MIN)  return "EARLY_DEP";
    return "ON_TIME";
  }
}

// 5-char status label for LCD row
const char* statusLabel(const String& st) {
  if (st == "LATE")      return "LATE ";
  if (st == "EARLY_ARR") return "E.ARR";
  if (st == "EARLY_DEP") return "E.DEP";
  if (st == "OVERTIME")  return "OT   ";
  return "OK   ";
}

// Hours worked as ≤5-char string (e.g. "8.0h ", "10.0h")
String calcHoursLcd(const String& inT, const String& outT) {
  if (inT.length() < 5 || outT.length() < 5) return "?h   ";
  int diff = (outT.substring(0, 2).toInt() * 60 + outT.substring(3, 5).toInt())
           - (inT.substring(0, 2).toInt()  * 60 + inT.substring(3, 5).toInt());
  if (diff <= 0) return "0h   ";
  char buf[8];
  snprintf(buf, sizeof(buf), "%.1fh", diff / 60.0f);
  String s = String(buf);
  while ((int)s.length() < 5) s += " ";
  return s.substring(0, 5);
}


// ════════════════════════════════════════════════════════════════
//  8. SPIFFS OFFLINE QUEUE
//     Format per line:  uid|date|intime|outtime|status
// ════════════════════════════════════════════════════════════════

void saveOffline(const String& uid, const String& date,
                 const String& inT,  const String& outT,
                 const String& status = "ON_TIME") {
  File f = SPIFFS.open("/queue.txt", FILE_APPEND);
  if (!f) { Serial.println("[SPIFFS] Open failed"); return; }
  f.println(uid + "|" + date + "|" +
            (inT.isEmpty()  ? "-" : inT)  + "|" +
            (outT.isEmpty() ? "-" : outT) + "|" +
            status);
  f.close();
  offlineCount++;
  Serial.printf("[SPIFFS] Saved offline #%d\n", offlineCount);
}


// ════════════════════════════════════════════════════════════════
//  9. HTTP (HTTPS GET to Google Apps Script)
// ════════════════════════════════════════════════════════════════

String httpGet(const String& url) {
  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  http.begin(client, url);
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  http.setTimeout(8000);
  int code = http.GET();
  String body = (code > 0) ? http.getString() : "";
  http.end();
  Serial.printf("[HTTP] %d\n", code);
  return body;
}

// Send an attendance log entry to GAS
String sendLog(const String& uid,    const String& date,
               const String& inT,    const String& outT,
               const String& status) {
  String url = String(GAS_URL) +
    "?action=log&uid="     + urlEncode(uid)    +
    "&date="               + urlEncode(date)   +
    "&intime="             + urlEncode(inT)    +
    "&outtime="            + urlEncode(outT)   +
    "&status="             + urlEncode(status);
  return httpGet(url);
}

// Push UID to server's pending queue (admin enrollment)
String pushAdminScan(const String& uid) {
  String url = String(GAS_URL) + "?action=pushScan&uid=" + urlEncode(uid);
  return httpGet(url);
}


// ════════════════════════════════════════════════════════════════
//  10. UPLOAD OFFLINE QUEUE (called when WiFi restores)
// ════════════════════════════════════════════════════════════════

void uploadOffline() {
  if (!SPIFFS.exists("/queue.txt")) return;
  Serial.println("[SPIFFS] Uploading offline records...");
  lcdMsg("Syncing offline ", "Please wait...  ");

  File f = SPIFFS.open("/queue.txt", FILE_READ);
  if (!f) return;

  File tmp = SPIFFS.open("/queue_tmp.txt", FILE_WRITE);
  int synced = 0, failed = 0;

  while (f.available()) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.isEmpty()) continue;

    // Parse uid|date|intime|outtime|status  (status optional for v4.0 records)
    String parts[5];
    int idx = 0, start = 0;
    for (int i = 0; i <= (int)line.length() && idx < 5; i++) {
      if (i == (int)line.length() || line[i] == '|') {
        parts[idx++] = line.substring(start, i);
        start = i + 1;
      }
    }
    if (idx < 4) { tmp.println(line); failed++; continue; } // malformed

    String inT    = (parts[2] == "-") ? "" : parts[2];
    String outT   = (parts[3] == "-") ? "" : parts[3];
    String status = (idx > 4 && parts[4].length() > 0) ? parts[4] : "ON_TIME";

    String resp = sendLog(parts[0], parts[1], inT, outT, status);

    StaticJsonDocument<256> doc;
    bool ok = false;
    if (deserializeJson(doc, resp) == DeserializationError::Ok) {
      ok = doc["ok"] | false;
    }

    if (ok) {
      synced++;
    } else {
      tmp.println(line);
      failed++;
    }
    delay(400);
  }

  f.close();
  tmp.close();

  SPIFFS.remove("/queue.txt");
  if (failed > 0) {
    SPIFFS.rename("/queue_tmp.txt", "/queue.txt");
  } else {
    SPIFFS.remove("/queue_tmp.txt");
  }

  offlineCount = failed;
  Serial.printf("[SPIFFS] Synced %d, retained %d\n", synced, failed);
  updateClock();
}


// ════════════════════════════════════════════════════════════════
//  11. RFID UID READER
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
//  12. NORMAL MODE — attendance scan
//
//  LCD row formats (all exactly 16 chars):
//    Check-in:   "IN  HH:MM  XXXXX"   XXXXX = OK   / LATE  / E.ARR
//                "Name (padded)    "
//    Check-out:  "OUT HH:MM  XXXXX"   XXXXX = 2.5h / OT    / E.DEP
//                "Name (padded)    "
//    Offline:    "IN  (Offline)   "
//                "HH:MM  Q:N      "
// ════════════════════════════════════════════════════════════════

void processNormal(const String& uid) {
  bool   isIn    = getStatus(uid);
  DateTime t     = rtc.now();
  int    h = t.hour(), m = t.minute();
  String date    = getDateStr();
  String timeNow = getTimeStr();
  String inT     = isIn ? timeNow : "";
  String outT    = isIn ? ""      : timeNow;
  String status  = determineStatus(isIn, h, m);

  // Show "Scanning..." while waiting for server
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Scanning card...");
  lcd.setCursor(0, 1); lcd.print(uid.substring(0, 16));

  // ── Offline path ────────────────────────────────────────────
  if (!wifiOk) {
    saveOffline(uid, date, inT, outT, status);
    if (isIn) { saveCheckInTime(uid, timeNow); beepIn(); }
    else beepOut();
    setStatus(uid, !isIn);

    char l1[17];
    snprintf(l1, sizeof(l1), "%-16s", (timeNow + "  Q:" + String(offlineCount)).c_str());
    showResult(isIn ? "IN  (Offline)   " : "OUT (Offline)   ", l1, 2500);
    return;
  }

  // ── Online path ─────────────────────────────────────────────
  String resp = sendLog(uid, date, inT, outT, status);

  // HTTP failure → treat as offline
  if (resp.isEmpty()) {
    saveOffline(uid, date, inT, outT, status);
    if (isIn) { saveCheckInTime(uid, timeNow); beepIn(); }
    else beepOut();
    setStatus(uid, !isIn);

    char l1[17];
    snprintf(l1, sizeof(l1), "%-16s", (timeNow + "  Q:" + String(offlineCount)).c_str());
    showResult(isIn ? "IN  (Offline)   " : "OUT (Offline)   ", l1, 2500);
    return;
  }

  // ── Parse server response ───────────────────────────────────
  bool   ok    = false;
  String name  = "";
  String event = isIn ? "IN" : "OUT";

  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, resp) == DeserializationError::Ok) {
    ok    = doc["ok"]    | false;
    name  = doc["name"]  | "";
    event = doc["event"] | (isIn ? "IN" : "OUT");
  }

  // Unknown card (not in roster)
  if (!ok || name.isEmpty()) {
    char l1[17];
    snprintf(l1, sizeof(l1), "%-16s", uid.substring(0, 16).c_str());
    showResult("X Unknown Card  ", l1, 2500);
    return;
  }

  // ── Success ─────────────────────────────────────────────────
  setStatus(uid, event == "OUT");  // if OUT just happened, next is IN

  char line0[17], line1[17];
  if (event == "IN") {
    saveCheckInTime(uid, timeNow);
    // "IN  HH:MM  XXXXX"  (4 + 5 + 2 + 5 = 16)
    snprintf(line0, sizeof(line0), "IN  %s  %s", timeNow.c_str(), statusLabel(status));
    snprintf(line1, sizeof(line1), "%-16s", name.substring(0, 16).c_str());
    beepIn();
  } else {
    // Show status label for OVERTIME/EARLY_DEP; hours worked otherwise
    String inStored = getCheckInTime(uid);
    String info = (status == "OVERTIME" || status == "EARLY_DEP")
      ? String(statusLabel(status))
      : calcHoursLcd(inStored, timeNow);
    // "OUT HH:MM  XXXXX"  (4 + 5 + 2 + 5 = 16)
    snprintf(line0, sizeof(line0), "OUT %s  %s", timeNow.c_str(), info.c_str());
    snprintf(line1, sizeof(line1), "%-16s", name.substring(0, 16).c_str());
    beepOut();
  }
  showResult(line0, line1, 2500);
}


// ════════════════════════════════════════════════════════════════
//  13. ADMIN MODE — push UID to pending queue on server
// ════════════════════════════════════════════════════════════════

void processAdmin(const String& uid) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Sending to web..");
  lcd.setCursor(0, 1); lcd.print(uid.substring(0, 16));

  if (!wifiOk) {
    showResult("No WiFi!        ", "Need WiFi 4 enrl", 2500);
    return;
  }

  String resp = pushAdminScan(uid);
  if (resp.isEmpty()) {
    showResult("Server Error    ", "Try again       ", 2500);
    return;
  }

  StaticJsonDocument<256> doc;
  bool   ok       = false;
  bool   existing = false;
  String name     = "";

  if (deserializeJson(doc, resp) == DeserializationError::Ok) {
    ok       = doc["ok"]       | false;
    existing = doc["existing"] | false;
    name     = doc["name"]     | "";
  }

  if (!ok) {
    showResult("Server Error    ", "Try again       ", 2500);
    return;
  }

  if (existing) {
    char l0[17];
    snprintf(l0, sizeof(l0), "%-16s", ("Enrolled: " + name).substring(0, 16).c_str());
    showResult(l0, "Scan another crd", 2500);
  } else {
    char l1[17];
    snprintf(l1, sizeof(l1), "%-16s", uid.substring(0, 16).c_str());
    showResult("Card Sent! ->   ", "Fill at web app ", 2500);
  }

  // Reset admin timeout on each successful scan
  adminModeMs = millis();
}


// ════════════════════════════════════════════════════════════════
//  SETUP
// ════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println(F("\n=== Smart RFID Attendance v4.1 ==="));

  // GPIO
  pinMode(BUZZER,    OUTPUT); digitalWrite(BUZZER, LOW);
  pinMode(ADMIN_BTN, INPUT_PULLUP);

  // I2C + LCD
  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();
  lcdMsg("Smart Attendance", ORG_NAME);
  delay(2000);

  // RTC
  if (!rtc.begin()) {
    lcdMsg("RTC Error!      ", "Check wiring    ");
    Serial.println("[RTC] Not found — check wiring and RTClib install");
    while (1) delay(1000);
  }
  if (rtc.lostPower()) {
    // Set to compile time when RTC battery is dead; user can adjust via serial later
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    Serial.println("[RTC] Lost power — reset to compile time");
  }
  Serial.printf("[RTC] %s %s\n", getDateStr().c_str(), getTimeStr().c_str());

  // SPIFFS
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
      Serial.printf("[SPIFFS] %d pending record(s)\n", offlineCount);
    }
  }

  // Preferences (per-card state)
  prefs.begin("attend", false);

  // RFID
  SPI.begin();
  rfid.PCD_Init();
  delay(50);
  Serial.println(F("[RFID] Ready"));

  // WiFi
  lcdMsg("Connecting WiFi.", WIFI_SSID);
  connectWiFi();

  // Sync any offline records immediately on boot
  if (wifiOk && offlineCount > 0) uploadOffline();

  updateClock();
  Serial.println(F("[SYS] Ready\n"));
}


// ════════════════════════════════════════════════════════════════
//  LOOP
// ════════════════════════════════════════════════════════════════

void loop() {
  unsigned long now = millis();

  // ── Admin mode: hold BOOT button for ADMIN_HOLD_MS ────────
  bool btnLow = (digitalRead(ADMIN_BTN) == LOW);
  if (btnLow && !btnWasLow) { btnWasLow = true; btnLowMs = now; }
  if (btnLow && btnWasLow && (now - btnLowMs >= ADMIN_HOLD_MS)) {
    adminMode   = !adminMode;
    adminModeMs = now;
    btnWasLow   = false;
    Serial.printf("[BTN] Admin mode %s\n", adminMode ? "ON" : "OFF");
    updateClock();
    delay(300);
  }
  if (!btnLow) btnWasLow = false;

  // ── Admin auto-timeout ─────────────────────────────────────
  if (adminMode && (now - adminModeMs > ADMIN_TIMEOUT_MS)) {
    adminMode = false;
    Serial.println("[BTN] Admin mode timed out");
    updateClock();
  }

  // ── WiFi watchdog ─────────────────────────────────────────
  if (now - lastWifiMs > WIFI_CHECK_MS) {
    lastWifiMs = now;
    bool wasOk = wifiOk;
    if (WiFi.status() != WL_CONNECTED) { wifiOk = false; connectWiFi(); }
    if (!wasOk && wifiOk) updateClock();  // refresh hint on reconnect
  }

  // ── Sync offline queue on WiFi restore ────────────────────
  if (wifiOk && offlineCount > 0) uploadOffline();

  // ── Clock update (1 s) ────────────────────────────────────
  if (now - lastClockMs > 1000) { lastClockMs = now; updateClock(); }

  // ── RFID poll ─────────────────────────────────────────────
  if (!rfid.PICC_IsNewCardPresent()) return;
  if (!rfid.PICC_ReadCardSerial())   return;

  String uid = readUID();

  // Debounce: ignore same card within SCAN_DEBOUNCE_MS
  if (uid == lastUID && (now - lastScanMs) < SCAN_DEBOUNCE_MS) {
    rfid.PICC_HaltA();
    return;
  }
  lastUID    = uid;
  lastScanMs = now;

  Serial.printf("[RFID] Card: %s  Admin:%d\n", uid.c_str(), adminMode);

  if (adminMode) processAdmin(uid);
  else           processNormal(uid);

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
}
