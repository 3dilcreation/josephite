#pragma once

// ── WiFi ─────────────────────────────────────────────────────────
#define WIFI_SSID      "TechSei_2.4G"
#define WIFI_PASSWORD  "92997080"

// ── Google Apps Script Web App URL ───────────────────────────────
// Deploy Code.gs → New Deployment → Web App → Anyone
#define GAS_URL  "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"

// ── RFID (MFRC522 via SPI) ────────────────────────────────────────
#define SS_PIN   5
#define RST_PIN  27

// ── I2C LCD 16×2 ─────────────────────────────────────────────────
#define LCD_ADDR  0x27   // try 0x3F if 0x27 doesn't work
#define SDA_PIN   21
#define SCL_PIN   22

// ── Buzzer (active buzzer — HIGH = ON) ───────────────────────────
// ONLY used for check-in and check-out confirmations
#define BUZZER  4

// ── Admin Mode Button ─────────────────────────────────────────────
// GPIO 0 = BOOT button on most ESP32 dev boards (no extra hardware)
// Hold 1 second to enter/exit admin mode
#define ADMIN_BTN           0
#define ADMIN_HOLD_MS    1000   // hold duration to trigger admin mode
#define ADMIN_TIMEOUT_MS 60000  // auto-exit after 60 s of inactivity

// ── Scan debounce ─────────────────────────────────────────────────
#define SCAN_DEBOUNCE_MS  4000  // ignore same card for 4 s

// ── WiFi reconnect interval ───────────────────────────────────────
#define WIFI_CHECK_MS  30000
