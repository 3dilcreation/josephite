#pragma once

// =============================================================
//  SMART RFID ATTENDANCE SYSTEM — Configuration
//  Edit this file before flashing to your ESP32
// =============================================================

// ----- WiFi -----
#define WIFI_SSID      "YOUR_WIFI_SSID"
#define WIFI_PASSWORD  "YOUR_WIFI_PASSWORD"

// ----- Google Apps Script Web App URL -----
// Deploy Code.gs as a Web App → "Anyone" access, paste the URL below
#define GAS_URL  "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"

// ----- Device Identity -----
#define DEVICE_NAME  "Main Entrance"   // Shown in attendance records
#define DEVICE_ID    "DEV_01"          // Unique device ID

// ----- I2C LCD -----
#define LCD_I2C_ADDR  0x27   // Try 0x3F if 0x27 doesn't work
#define LCD_COLS      16
#define LCD_ROWS      2
#define SDA_PIN       21
#define SCL_PIN       22

// ----- Hardware Pins -----
#define SS_PIN        5     // RFID SDA/SS
#define RST_PIN       4     // RFID RST
#define BUZZER_PIN    2     // Buzzer (passive recommended)
#define LED_GREEN_PIN 15    // Green indicator LED
#define LED_RED_PIN   16    // Red indicator LED

// ----- Buzzer type -----
// Set to 1 for PASSIVE buzzer (supports tones), 0 for ACTIVE buzzer
#define PASSIVE_BUZZER  1

// ----- Timezone -----
// UTC offset in seconds  (Philippines = +8 → 28800)
#define GMT_OFFSET_SEC   28800
#define DAYLIGHT_SEC     0

// ----- Attendance Rules (mirrors Settings sheet) -----
// Used on-device for real-time LCD feedback only.
// The server is the source of truth for records.
#define CHECKIN_START_HOUR   8    // Attendance window opens 08:00
#define LATE_HOUR            9    // After 09:00 → "LATE"
#define LATE_MINUTE          0
#define EXPECTED_OUT_HOUR   17    // Expected checkout 17:00
#define OVERTIME_HOUR       18    // After 18:00 → "OVERTIME"

// ----- Debounce (ms) — prevents duplicate scans -----
#define DEBOUNCE_MS          2000

// ----- Offline queue size (max records stored in flash) -----
#define MAX_OFFLINE_QUEUE    50

// ----- Auto re-check WiFi every N milliseconds -----
#define WIFI_CHECK_INTERVAL  30000

// ----- LCD clock refresh interval (ms) -----
#define CLOCK_REFRESH_MS     1000
