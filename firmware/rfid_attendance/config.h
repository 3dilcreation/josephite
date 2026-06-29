#pragma once

// ── Organization ──────────────────────────────────────────────
#define ORG_NAME       "TechSei Lab    "  // exactly 16 chars for LCD splash

// ── WiFi ─────────────────────────────────────────────────────
#define WIFI_SSID      "TechSei_2.4G"
#define WIFI_PASSWORD  "92997080"

// ── Google Apps Script Web App URL ───────────────────────────
// Deploy Code.gs → New Deployment → Web App → Execute as Me → Anyone
#define GAS_URL  "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"

// ── RFID (MFRC522 via SPI) ────────────────────────────────────
#define SS_PIN   5
#define RST_PIN  27

// ── I2C LCD 16×2 ─────────────────────────────────────────────
#define LCD_ADDR  0x27   // try 0x3F if 0x27 doesn't work
#define SDA_PIN   21
#define SCL_PIN   22

// ── Buzzer (active buzzer — HIGH = ON) ───────────────────────
// ONLY used for check-in and check-out confirmations
#define BUZZER  4

// ── Admin Mode Button ─────────────────────────────────────────
// GPIO 0 = BOOT button on most ESP32 dev boards (no extra hardware needed)
// Hold for ADMIN_HOLD_MS to enter/exit admin mode
#define ADMIN_BTN           0
#define ADMIN_HOLD_MS    1000   // ms to hold button to toggle admin mode
#define ADMIN_TIMEOUT_MS 60000  // auto-exit admin mode after 60 s inactivity

// ── Scan debounce ─────────────────────────────────────────────
#define SCAN_DEBOUNCE_MS  4000  // ignore same card within 4 s

// ── WiFi reconnect interval ───────────────────────────────────
#define WIFI_CHECK_MS  30000    // check WiFi every 30 s

// ── Attendance Time Thresholds ────────────────────────────────
// All times are 24-hour format (0–23 for hours, 0–59 for minutes)

// Earliest valid check-in — earlier scans flagged EARLY_ARR
#define OPEN_HOUR        7
#define OPEN_MIN         0

// Late threshold — check-in after this time is flagged LATE
#define LATE_HOUR        9
#define LATE_MIN         0

// Latest valid check-in / system closes after this
#define CLOSE_HOUR      20
#define CLOSE_MIN        0

// Early departure — check-out BEFORE this time is flagged EARLY_DEP
#define EARLY_OUT_HOUR  16
#define EARLY_OUT_MIN    0

// Overtime — check-out AFTER this time is flagged OVERTIME
#define OVERTIME_HOUR   18
#define OVERTIME_MIN     0
