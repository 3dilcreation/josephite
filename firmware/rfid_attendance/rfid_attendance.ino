/*
 * ================================================================
 *   Smart RFID Attendance System  v4.0  —  TechSei Lab
 * ================================================================
 *   Hardware:
 *     ESP32  |  MFRC522 RFID  |  DS3231 RTC  |  I2C LCD 16×2
 *     Active buzzer on GPIO 4  |  BOOT button on GPIO 0
 *
 *   Libraries  (Arduino Library Manager):
 *     MFRC522           by GithubCommunity
 *     LiquidCrystal_I2C by Frank de Brabander
 *     RTClib            by Adafruit
 *     ArduinoJson       by Benoit Blanchon  v6.x
 *
 *   Features:
 *     • Check-in / check-out toggle per UID (Preferences, survives reboot)
 *     • DS3231 RTC — no internet time needed
 *     • SPIFFS offline queue — auto-uploads when WiFi returns
 *     • Buzzer ONLY on check-in / check-out confirmation
 *     • Admin mode (hold BOOT 1 s) — scan card → push UID to server
 *       → enroll name/dept/ID via web dashboard
 *     • Student / Staff / Others support (matches GAS Roster sheet)
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
Preferences       prefs;       // IN/OUT state per card UID

// ── Global state ─────────────────────────────────────────────────
bool          wifiOk       = false;
bool          adminMode    = false;
unsigned long adminModeMs  = 0;   // millis() when admin mode started
String        lastUID      = "";
unsigned long lastScanMs   = 0;
unsigned long lastWifiMs   = 0;
unsigned long lastClockMs  = 0;
bool          btnWasLow    = false;
unsigned long btnLowMs     = 0;
int           offlineCount = 0;

const char* DAYS[] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};


// ════════════════════════════════════════════════════════════════
//  1. BUZZER  — only IN and OUT confirmations
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
  char row0[17], row1[17];

  snprintf(row0, sizeof(row0), "%s %02d/%02d  %02d:%02d",
    DAYS[now.dayOfTheWeek()], now.day(), now.month(),
    now.hour(), now.minute());

  if (adminMode) {
    snprintf(row1, sizeof(row1), "**ADMIN** Scan  ");
  } else if (wifiOk) {
    snprintf(row1, sizeof(row1), "[WiFi] Scan Card");
  } else {
    snprintf(row1, sizeof(row1), "[OFLN] Scan Card");
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
//  3. URL ENCODE  (needed for GAS GET requests)
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
//  6. PREFERENCES — IN/OUT status per card UID
// ════════════════════════════════════════════════════════════════

// Compress UID to ≤15-char Preferences key (remove colons, take first 15)
String makeKey(const String& uid) {
  String k = uid;
  k.replace(":", "");
  k.replace(" ", "");
  if (k.length() > 15) k = k.substring(0, 15);
  return k;
}

bool getStatus(const String& uid) {
  // true  = next scan is IN
  // false = next scan is OUT
  return prefs.getBool(makeKey(uid).c_str(), true);
}

void setStatus(const String& uid, bool nextIsIn) {
  prefs.putBool(makeKey(uid).c_str(), nextIsIn);
}


// ════════════════════════════════════════════════════════════════
//  7. SPIFFS OFFLINE QUEUE
//     Format per line:  uid|date|intime|outtime
//     Server fills name from Roster on upload
// ════════════════════════════════════════════════════════════════

void saveOffline(const String& uid, const String& date,
                 const String& inT, const String& outT) {
  File f = SPIFFS.open("/queue.txt", FILE_APPEND);
  if (!f) { Serial.println("[SPIFFS] Open failed"); return; }
  f.println(uid + "|" + date + "|" +
            (inT.isEmpty()  ? "-" : inT)  + "|" +
            (outT.isEmpty() ? "-" : outT));
  f.close();
  offlineCount++;
  Serial.printf("[SPIFFS] Saved offline #%d\n", offlineCount);
}

// Returns HTTP response body, or "" on failure
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

void uploadOffline() {
  if (!SPIFFS.exists("/queue.txt")) return;
  Serial.println("[SPIFFS] Uploading offline records...");
  lcdMsg("Syncing offline", "Please wait...");

  File f = SPIFFS.open("/queue.txt", FILE_READ);
  if (!f) return;

  File tmp = SPIFFS.open("/queue_tmp.txt", FILE_WRITE);
  int synced = 0, failed = 0;

  while (f.available()) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line.isEmpty()) continue;

    // Parse uid|date|intime|outtime
    String parts[4];
    int idx = 0, start = 0;
    for (int i = 0; i <= (int)line.length() && idx < 4; i++) {
      if (i == (int)line.length() || line[i] == '|') {
        parts[idx++] = line.substring(start, i);
        start = i + 1;
      }
    }
    String inT  = (parts[2] == "-") ? "" : parts[2];
    String outT = (parts[3] == "-") ? "" : parts[3];

    String url = String(GAS_URL) +
      "?action=log&uid=" + urlEncode(parts[0]) +
      "&date=" + urlEncode(parts[1]) +
      "&intime=" + urlEncode(inT) +
      "&outtime=" + urlEncode(outT);

    String resp = httpGet(url);
    StaticJsonDocument<256> doc;
    bool ok = false;
    if (deserializeJson(doc, resp) == DeserializationError::Ok) {
      ok = doc["ok"] | false;
    }

    if (ok) {
      synced++;
    } else {
      // Keep failed lines for next attempt
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
  Serial.printf("[SPIFFS] Synced %d, retained %d failed\n", synced, failed);
  updateClock();
}


// ════════════════════════════════════════════════════════════════
//  8. GAS HTTP CALLS
// ════════════════════════════════════════════════════════════════

// Log attendance — returns JSON {ok, event, name, type, late}
String logAttendance(const String& uid, const String& date,
                     const String& inT, const String& outT) {
  String url = String(GAS_URL) +
    "?action=log&uid=" + urlEncode(uid) +
    "&date=" + urlEncode(date) +
    "&intime=" + urlEncode(inT) +
    "&outtime=" + urlEncode(outT);
  return httpGet(url);
}

// Push UID to server pending queue (admin mode)
// Returns {ok, existing, name} — existing=true if already in roster
String pushAdminScan(const String& uid) {
  String url = String(GAS_URL) +
    "?action=pushScan&uid=" + urlEncode(uid);
  return httpGet(url);
}


// ════════════════════════════════════════════════════════════════
//  9. RFID UID READER
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
//  10. CARD PROCESSING — NORMAL MODE
// ════════════════════════════════════════════════════════════════

void processNormal(const String& uid) {
  bool isIn     = getStatus(uid);   // true = this scan is a check-in
  String date   = getDateStr();
  String timeNow = getTimeStr();
  String inT    = isIn  ? timeNow : "";
  String outT   = isIn  ? ""      : timeNow;

  // Show "Scanning..." while waiting for server
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Scanning card...");
  lcd.setCursor(0, 1); lcd.print(uid.substring(0, 16));

  String resp = "";
  bool   online = wifiOk;

  if (online) resp = logAttendance(uid, date, inT, outT);

  // ── Parse server response ──────────────────────────────────
  bool   ok      = false;
  String name    = "";
  String event   = isIn ? "IN" : "OUT";
  bool   late    = false;

  if (!resp.isEmpty()) {
    StaticJsonDocument<256> doc;
    if (deserializeJson(doc, resp) == DeserializationError::Ok) {
      ok    = doc["ok"]    | false;
      name  = doc["name"]  | "";
      event = doc["event"] | (isIn ? "IN" : "OUT");
      late  = doc["late"]  | false;
    }
  }

  if (!online || !ok) {
    // Save to SPIFFS and show offline message
    saveOffline(uid, date, inT, outT);
    // Still toggle and beep — user experience matters
    String l0 = isIn ? "IN  (Offline)" : "OUT (Offline)";
    String l1 = timeNow + "  Queue:" + String(offlineCount);
    if (isIn) beepIn(); else beepOut();
    setStatus(uid, !isIn);
    showResult(l0.c_str(), l1.c_str(), 2500);
    return;
  }

  if (!ok || name.isEmpty()) {
    // Unknown card
    char l1[17];
    snprintf(l1, sizeof(l1), "%-16s", uid.substring(0,16).c_str());
    showResult("X Unknown Card  ", l1, 2500);
    return;
  }

  // ── Success ───────────────────────────────────────────────
  setStatus(uid, event == "OUT");   // if OUT just happened, next is IN

  char line0[17], line1[17];
  if (event == "IN") {
    snprintf(line0, sizeof(line0), "IN   %-11s", name.substring(0,11).c_str());
    snprintf(line1, sizeof(line1), "%s%s", timeNow.c_str(), late ? "  *LATE*" : "         ");
    beepIn();
  } else {
    snprintf(line0, sizeof(line0), "OUT  %-11s", name.substring(0,11).c_str());
    snprintf(line1, sizeof(line1), "%s  Goodbye!", timeNow.c_str());
    beepOut();
  }
  showResult(line0, line1, 2500);
}


// ════════════════════════════════════════════════════════════════
//  11. CARD PROCESSING — ADMIN MODE
// ════════════════════════════════════════════════════════════════

void processAdmin(const String& uid) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Sending to web..");
  lcd.setCursor(0, 1); lcd.print(uid.substring(0, 16));

  String resp = "";
  if (wifiOk) resp = pushAdminScan(uid);

  if (!wifiOk || resp.isEmpty()) {
    showResult("No WiFi!        ", "Need WiFi 4 enrl", 2500);
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
    snprintf(l0, sizeof(l0), "%-16s", ("Enrolled: " + name).substring(0,16).c_str());
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
  Serial.println(F("\n=== Smart RFID Attendance v4.0 ==="));

  // Pins
  pinMode(BUZZER,    OUTPUT); digitalWrite(BUZZER, LOW);
  pinMode(ADMIN_BTN, INPUT_PULLUP);

  // I2C + LCD
  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();

  // Splash
  lcdMsg("Smart Attendance", "  TechSei v4.0  ");
  delay(2000);

  // RTC
  if (!rtc.begin()) {
    lcdMsg("RTC Error!", "Check wiring    ");
    Serial.println("[RTC] Not found!");
    while (1) delay(1000);
  }
  if (rtc.lostPower()) {
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
    Serial.println("[RTC] Time reset to compile time");
  }
  Serial.printf("[RTC] %s\n", getDateStr().c_str());

  // SPIFFS
  if (!SPIFFS.begin(true)) {
    Serial.println("[SPIFFS] Mount failed");
  } else {
    // Count pending offline records
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

  // Preferences
  prefs.begin("attend", false);

  // RFID
  SPI.begin();
  rfid.PCD_Init();
  delay(50);
  Serial.println(F("[RFID] Ready"));

  // WiFi
  lcdMsg("Connecting WiFi.", WIFI_SSID);
  connectWiFi();

  // Upload any offline records immediately
  if (wifiOk && offlineCount > 0) uploadOffline();

  updateClock();
  Serial.println(F("[SYS] Ready\n"));
}


// ════════════════════════════════════════════════════════════════
//  LOOP
// ════════════════════════════════════════════════════════════════

void loop() {
  unsigned long now = millis();

  // ── Admin mode button (hold BOOT/GPIO 0 for 1 s) ──────────
  bool btnLow = (digitalRead(ADMIN_BTN) == LOW);
  if (btnLow && !btnWasLow) {
    btnWasLow = true;
    btnLowMs  = now;
  }
  if (btnLow && btnWasLow && (now - btnLowMs >= ADMIN_HOLD_MS)) {
    // Toggle admin mode
    adminMode   = !adminMode;
    adminModeMs = now;
    btnWasLow   = false;  // consume this press
    Serial.printf("[BTN] Admin mode %s\n", adminMode ? "ON" : "OFF");
    updateClock();
    delay(300);  // small debounce after toggle
  }
  if (!btnLow) btnWasLow = false;

  // ── Admin mode auto-timeout ────────────────────────────────
  if (adminMode && (now - adminModeMs > ADMIN_TIMEOUT_MS)) {
    adminMode = false;
    Serial.println("[BTN] Admin mode timed out");
    updateClock();
  }

  // ── WiFi watchdog ─────────────────────────────────────────
  if (now - lastWifiMs > WIFI_CHECK_MS) {
    lastWifiMs = now;
    if (WiFi.status() != WL_CONNECTED) {
      wifiOk = false;
      connectWiFi();
    }
  }

  // ── Sync offline queue when WiFi restores ─────────────────
  if (wifiOk && offlineCount > 0) {
    uploadOffline();
  }

  // ── Live clock update every second ────────────────────────
  if (now - lastClockMs > 1000) {
    lastClockMs = now;
    updateClock();
  }

  // ── RFID poll ─────────────────────────────────────────────
  if (!rfid.PICC_IsNewCardPresent()) return;
  if (!rfid.PICC_ReadCardSerial())   return;

  String uid = readUID();

  // Debounce: ignore same card within 4 s
  if (uid == lastUID && (now - lastScanMs) < SCAN_DEBOUNCE_MS) {
    rfid.PICC_HaltA();
    return;
  }
  lastUID   = uid;
  lastScanMs = now;

  Serial.printf("[RFID] Card: %s  Admin:%d\n", uid.c_str(), adminMode);

  if (adminMode) {
    processAdmin(uid);
  } else {
    processNormal(uid);
  }

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
}
