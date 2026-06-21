/*
 * ================================================================
 *   SMART RFID ATTENDANCE SYSTEM — ESP32 Firmware
 *   Features:
 *     • MFRC522 RFID reader (SPI)
 *     • I2C LCD 16×2 with custom characters & live clock
 *     • Distinct buzzer tones for check-in / check-out / error
 *     • Automatic check-in ↔ check-out toggle via Google Sheets
 *     • NTP time sync (no RTC needed)
 *     • Offline queue in flash — auto-syncs when WiFi returns
 *     • Late-arrival & status indicator on LCD
 *     • Green/Red LED feedback
 *     • WiFi watchdog with auto-reconnect
 *
 *  Required libraries (install via Arduino Library Manager):
 *     MFRC522       — by GithubCommunity
 *     LiquidCrystal_I2C — by Frank de Brabander
 *     ArduinoJson   — by Benoit Blanchon (v6.x)
 * ================================================================
 */

#include <SPI.h>
#include <MFRC522.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <time.h>
#include "config.h"

// ── Objects ──────────────────────────────────────────────────────
MFRC522           rfid(SS_PIN, RST_PIN);
LiquidCrystal_I2C lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS);
Preferences       nvs;

// ── Custom LCD glyphs ────────────────────────────────────────────
byte gCheck[8]  = {0b00000,0b00001,0b00011,0b10110,0b11100,0b01000,0b00000,0b00000};
byte gCross[8]  = {0b00000,0b10001,0b01010,0b00100,0b01010,0b10001,0b00000,0b00000};
byte gWifi[8]   = {0b00000,0b01110,0b10001,0b00100,0b01010,0b00000,0b00100,0b00000};
byte gBell[8]   = {0b00100,0b01110,0b01110,0b01110,0b11111,0b00000,0b00100,0b00000};
byte gClock[8]  = {0b00000,0b01110,0b10101,0b10111,0b10001,0b01110,0b00000,0b00000};
byte gLock[8]   = {0b01110,0b10001,0b10001,0b11111,0b11011,0b11011,0b11111,0b00000};
byte gArUp[8]   = {0b00100,0b01110,0b11111,0b00100,0b00100,0b00100,0b00100,0b00000};
byte gArDn[8]   = {0b00100,0b00100,0b00100,0b00100,0b11111,0b01110,0b00100,0b00000};

// ── State ────────────────────────────────────────────────────────
bool     wifiOk          = false;
bool     timeSynced      = false;
bool     processing      = false;
int      offlineCount    = 0;
String   lastUID         = "";
unsigned long lastScanMs = 0;
unsigned long lastWifiCheck = 0;
unsigned long lastClockMs   = 0;

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  Serial.println(F("\n╔══════════════════════════════╗"));
  Serial.println(F("║  Smart RFID Attendance v2.0  ║"));
  Serial.println(F("╚══════════════════════════════╝"));

  // GPIO
  pinMode(BUZZER_PIN,    OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_RED_PIN,   OUTPUT);
  setLed(false);

  // LCD
  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();
  lcd.createChar(0, gCheck);
  lcd.createChar(1, gCross);
  lcd.createChar(2, gWifi);
  lcd.createChar(3, gBell);
  lcd.createChar(4, gClock);
  lcd.createChar(5, gLock);
  lcd.createChar(6, gArUp);
  lcd.createChar(7, gArDn);
  splashScreen();

  // SPI + RFID
  SPI.begin();
  rfid.PCD_Init();
  delay(50);

  // NVS (offline queue)
  nvs.begin("attendance", false);
  offlineCount = nvs.getInt("q_size", 0);
  if (offlineCount > 0) {
    Serial.printf("[NVS] %d offline record(s) pending sync\n", offlineCount);
  }

  // WiFi
  lcdMsg("Connecting WiFi", "Please wait...");
  connectWiFi();

  // NTP time sync
  if (wifiOk) {
    lcdMsg("Syncing time...", "NTP pool.ntp.org");
    configTime(GMT_OFFSET_SEC, DAYLIGHT_SEC, "pool.ntp.org", "time.nist.gov");
    waitNTP();
  }

  displayReady();
  Serial.println(F("[SYS] Ready\n"));
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();

  // ── WiFi watchdog ──
  if (now - lastWifiCheck > WIFI_CHECK_INTERVAL) {
    lastWifiCheck = now;
    if (WiFi.status() != WL_CONNECTED) {
      wifiOk = false;
      setLed(false);
      connectWiFi();
    }
  }

  // ── Sync offline queue on WiFi restore ──
  if (wifiOk && offlineCount > 0) {
    syncOfflineQueue();
  }

  // ── Live clock on LCD (idle only) ──
  if (!processing && now - lastClockMs > CLOCK_REFRESH_MS) {
    lastClockMs = now;
    updateClock();
  }

  // ── RFID poll ──
  if (!rfid.PICC_IsNewCardPresent()) return;
  if (!rfid.PICC_ReadCardSerial())   return;

  String uid = readUID();

  // Debounce
  if (uid == lastUID && (now - lastScanMs) < DEBOUNCE_MS) {
    rfid.PICC_HaltA();
    return;
  }
  lastUID    = uid;
  lastScanMs = now;

  Serial.printf("[RFID] Card: %s\n", uid.c_str());
  processCard(uid);

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
}

// ═══════════════════════════════════════════════════════════════
//  CARD PROCESSING
// ═══════════════════════════════════════════════════════════════
void processCard(const String& uid) {
  processing = true;

  struct tm ti;
  if (!getLocalTime(&ti)) {
    lcdMsg("\x01 Time Error!", "Reconnect WiFi");
    beepError();
    delay(2000);
    displayReady();
    processing = false;
    return;
  }

  // Build ISO timestamp
  char ts[25];
  strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &ti);

  // Local late detection for LCD hint
  bool localLate = (ti.tm_hour > LATE_HOUR) ||
                   (ti.tm_hour == LATE_HOUR && ti.tm_min >= LATE_MINUTE);

  // JSON payload
  StaticJsonDocument<256> doc;
  doc["uid"]       = uid;
  doc["timestamp"] = ts;
  doc["device"]    = DEVICE_NAME;
  doc["device_id"] = DEVICE_ID;

  String payload;
  serializeJson(doc, payload);

  // Animate LCD
  lcdMsg("Reading card...", uid.length() > 16 ? uid.substring(0,16) : uid);
  delay(300);

  if (wifiOk) {
    postRecord(payload);
  } else {
    queueOffline(payload);
    lcdIcon(1, "Saved (offline)", "Queue:" + String(offlineCount));
    beepWarn();
    delay(2200);
    displayReady();
  }

  processing = false;
}

// ═══════════════════════════════════════════════════════════════
//  GOOGLE SHEETS POST
// ═══════════════════════════════════════════════════════════════
void postRecord(const String& payload, bool isSync) {
  WiFiClientSecure client;
  client.setInsecure();  // Skip cert verification (GAS uses valid cert anyway)

  HTTPClient https;
  https.begin(client, GAS_URL);
  https.addHeader("Content-Type", "application/json");
  https.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  https.setTimeout(10000);

  Serial.printf("[HTTP] POST → %d bytes\n", payload.length());
  int code = https.POST(payload);
  String body = (code > 0) ? https.getString() : "";
  https.end();

  Serial.printf("[HTTP] Response %d: %s\n", code, body.c_str());

  if (code <= 0) {
    Serial.printf("[HTTP] Error: %s\n", HTTPClient::errorToString(code).c_str());
    if (!isSync) { queueOffline(payload); }
    lcdIcon(1, "Server Error", "Saved offline");
    beepError();
    delay(2200);
    displayReady();
    return;
  }

  // Parse response JSON
  StaticJsonDocument<512> resp;
  DeserializationError err = deserializeJson(resp, body);

  if (err || !resp.containsKey("status")) {
    // Likely redirect page — treat as success
    lcdIcon(0, "Recorded!", "");
    beepCheckIn();
    delay(2000);
    displayReady();
    return;
  }

  String status = resp["status"].as<String>();

  if (status == "success") {
    String name       = resp["name"]      | "Unknown";
    String action     = resp["action"]    | "CHECK_IN";
    bool   late       = resp["late"]      | false;
    bool   overtime   = resp["overtime"]  | false;
    String note       = resp["statusNote"]| "";

    showAttendanceResult(name, action, late, overtime, note);

  } else if (status == "unknown") {
    String uid = resp["uid"] | "?";
    lcdIcon(1, "Unknown Card!", uid.substring(0, 16));
    beepError();
    setLed(false);
    digitalWrite(LED_RED_PIN, HIGH);
    delay(2200);
    digitalWrite(LED_RED_PIN, LOW);
    displayReady();

  } else {
    String msg = resp["message"] | "Error";
    lcdIcon(1, "Error:", msg.substring(0, 16));
    beepError();
    delay(2200);
    displayReady();
  }
}

void showAttendanceResult(String name, String action, bool late, bool overtime, String note) {
  if (name.length() > 14) name = name.substring(0, 13) + ".";

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(byte(0));  // checkmark
  lcd.print(" ");
  lcd.print(name);

  lcd.setCursor(0, 1);
  if (action == "CHECK_IN") {
    lcd.write(byte(6));  // arrow up
    if (late)     lcd.print(" IN  *LATE*");
    else          lcd.print(" IN  OnTime");
    beepCheckIn();
  } else if (action == "CHECK_OUT") {
    lcd.write(byte(7));  // arrow down
    if (overtime) lcd.print(" OUT Overtime");
    else          lcd.print(" OUT Goodbye!");
    beepCheckOut();
  } else {
    lcd.print(action.substring(0, 16));
    beepCheckIn();
  }

  setLed(true);
  delay(2200);
  setLed(false);
  displayReady();
}

// ═══════════════════════════════════════════════════════════════
//  OFFLINE QUEUE  (NVS / Preferences)
// ═══════════════════════════════════════════════════════════════
void queueOffline(const String& payload) {
  if (offlineCount >= MAX_OFFLINE_QUEUE) {
    Serial.println(F("[NVS] Queue full, dropping oldest record"));
    // Shift queue (drop index 0, rename 1→0, 2→1, ...)
    for (int i = 0; i < MAX_OFFLINE_QUEUE - 1; i++) {
      String v = nvs.getString(("r" + String(i + 1)).c_str(), "");
      nvs.putString(("r" + String(i)).c_str(), v);
    }
    offlineCount = MAX_OFFLINE_QUEUE - 1;
  }
  nvs.putString(("r" + String(offlineCount)).c_str(), payload);
  offlineCount++;
  nvs.putInt("q_size", offlineCount);
  Serial.printf("[NVS] Queued record #%d\n", offlineCount);
}

void syncOfflineQueue() {
  Serial.printf("[SYNC] Syncing %d offline record(s)...\n", offlineCount);
  lcdMsg("Syncing records", String(offlineCount) + " queued...");

  int synced = 0;
  for (int i = 0; i < offlineCount; i++) {
    String key  = "r" + String(i);
    String data = nvs.getString(key.c_str(), "");
    if (data.length() == 0) continue;

    // Stamp as offline sync
    StaticJsonDocument<300> doc;
    if (deserializeJson(doc, data) == DeserializationError::Ok) {
      doc["offline_sync"] = true;
      serializeJson(doc, data);
    }

    postRecord(data, true);
    nvs.remove(key.c_str());
    synced++;
    delay(500);  // polite gap between requests
  }

  offlineCount = 0;
  nvs.putInt("q_size", 0);
  Serial.printf("[SYNC] Done — synced %d record(s)\n", synced);
  displayReady();
}

// ═══════════════════════════════════════════════════════════════
//  WIFI
// ═══════════════════════════════════════════════════════════════
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) { wifiOk = true; return; }

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);

  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiOk = true;
    Serial.printf("\n[WiFi] Connected — IP: %s\n", WiFi.localIP().toString().c_str());
    digitalWrite(LED_GREEN_PIN, HIGH);
    delay(100);
    digitalWrite(LED_GREEN_PIN, LOW);
  } else {
    wifiOk = false;
    Serial.println(F("\n[WiFi] Failed — Offline mode active"));
  }
}

// ═══════════════════════════════════════════════════════════════
//  NTP
// ═══════════════════════════════════════════════════════════════
void waitNTP() {
  struct tm ti;
  int tries = 0;
  while (!getLocalTime(&ti) && tries++ < 20) delay(500);
  if (tries < 20) {
    timeSynced = true;
    char buf[20];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", &ti);
    Serial.printf("[NTP] Time synced: %s\n", buf);
  } else {
    Serial.println(F("[NTP] Sync failed"));
  }
}

// ═══════════════════════════════════════════════════════════════
//  LCD HELPERS
// ═══════════════════════════════════════════════════════════════
void splashScreen() {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(" Smart Attend  ");
  lcd.setCursor(0, 1); lcd.print("  RFID v2.0   ");
  for (int i = 0; i < 3; i++) { beepBoot(); delay(150); }
  delay(1500);
}

void displayReady() {
  updateClock();
}

void updateClock() {
  struct tm ti;
  lcd.setCursor(0, 0);
  if (getLocalTime(&ti)) {
    char row0[17], row1[17];
    strftime(row0, sizeof(row0), "%a %b %d %H:%M", &ti);
    lcd.print(row0);
    lcd.setCursor(0, 1);
    if (wifiOk) { lcd.write(byte(2)); lcd.print(" Scan RFID Card"); }
    else         { lcd.write(byte(1)); lcd.print(" OFFLINE  Mode "); }
  } else {
    lcd.print("--:-- Scan Card ");
    lcd.setCursor(0, 1);
    lcd.print(wifiOk ? "\x02 Waiting NTP.." : "\x01 WiFi offline ");
  }
}

void lcdMsg(const String& top, const String& bot) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(top.substring(0, LCD_COLS));
  lcd.setCursor(0, 1); lcd.print(bot.substring(0, LCD_COLS));
}

// icon = 0 (check) or 1 (cross)
void lcdIcon(uint8_t icon, const String& top, const String& bot) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(byte(icon));
  lcd.print(" ");
  String t = top; if (t.length() > 14) t = t.substring(0, 14);
  lcd.print(t);
  lcd.setCursor(0, 1);
  lcd.print(bot.substring(0, LCD_COLS));
}

// ═══════════════════════════════════════════════════════════════
//  LED
// ═══════════════════════════════════════════════════════════════
void setLed(bool green) {
  digitalWrite(LED_GREEN_PIN, green ? HIGH : LOW);
  digitalWrite(LED_RED_PIN,   green ? LOW  : LOW);
}

// ═══════════════════════════════════════════════════════════════
//  BUZZER PATTERNS
// ═══════════════════════════════════════════════════════════════
void beepBoot() {
#if PASSIVE_BUZZER
  tone(BUZZER_PIN, 1000, 80); delay(100); noTone(BUZZER_PIN);
#else
  digitalWrite(BUZZER_PIN, HIGH); delay(80); digitalWrite(BUZZER_PIN, LOW);
#endif
}

void beepCheckIn() {
  // Rising two-note = welcome
#if PASSIVE_BUZZER
  tone(BUZZER_PIN, 880,  120); delay(140);
  tone(BUZZER_PIN, 1320, 180); delay(200);
  noTone(BUZZER_PIN);
#else
  for (int i = 0; i < 2; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(150);
    digitalWrite(BUZZER_PIN, LOW);  delay(80);
  }
#endif
}

void beepCheckOut() {
  // Falling two-note = goodbye
#if PASSIVE_BUZZER
  tone(BUZZER_PIN, 1320, 120); delay(140);
  tone(BUZZER_PIN, 880,  180); delay(200);
  noTone(BUZZER_PIN);
#else
  digitalWrite(BUZZER_PIN, HIGH); delay(150);
  digitalWrite(BUZZER_PIN, LOW);  delay(60);
  digitalWrite(BUZZER_PIN, HIGH); delay(80);
  digitalWrite(BUZZER_PIN, LOW);
#endif
}

void beepError() {
  // Low double-buzz
#if PASSIVE_BUZZER
  tone(BUZZER_PIN, 330, 250); delay(300);
  tone(BUZZER_PIN, 330, 250); delay(300);
  noTone(BUZZER_PIN);
#else
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH); delay(150);
    digitalWrite(BUZZER_PIN, LOW);  delay(100);
  }
#endif
}

void beepWarn() {
  // Single mid-pitch
#if PASSIVE_BUZZER
  tone(BUZZER_PIN, 660, 200); delay(250); noTone(BUZZER_PIN);
#else
  digitalWrite(BUZZER_PIN, HIGH); delay(200); digitalWrite(BUZZER_PIN, LOW);
#endif
}

// ═══════════════════════════════════════════════════════════════
//  UID READER
// ═══════════════════════════════════════════════════════════════
String readUID() {
  String uid = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) uid += "0";
    uid += String(rfid.uid.uidByte[i], HEX);
    if (i < rfid.uid.size - 1) uid += ":";
  }
  uid.toUpperCase();
  return uid;
}
