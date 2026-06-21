# Smart RFID Attendance System

ESP32 + MFRC522 + I2C LCD + Google Sheets — fully automated attendance tracking with live dashboard.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ESP32 Device                          │
│                                                         │
│  MFRC522 ──SPI──► ESP32 ──HTTPS POST──► Google Apps    │
│  I2C LCD ──I2C──► ESP32               Script (doPost)  │
│  Buzzer  ──GPIO──► ESP32                    │           │
│  LEDs    ──GPIO──► ESP32              Google Sheets     │
│                                             │           │
│  Offline queue                        HTML Dashboard    │
│  (NVS flash)                          (doGet → browser) │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware Wiring

### ESP32 ↔ MFRC522 (SPI)

```
ESP32 Pin   MFRC522 Pin    Wire colour (suggestion)
─────────   ───────────    ────────────────────────
GPIO 5      SDA / SS       Orange
GPIO 18     SCK            Yellow
GPIO 23     MOSI           Green
GPIO 19     MISO           Blue
GPIO 4      RST            White
3.3 V       3.3V           Red     ← MUST be 3.3 V, NOT 5 V!
GND         GND            Black
```

### ESP32 ↔ I2C LCD 16×2

```
ESP32 Pin   LCD Pin    Notes
─────────   ───────    ──────────────────────────────
GPIO 21     SDA        I2C data
GPIO 22     SCL        I2C clock
5 V         VCC        LCD needs 5 V
GND         GND
```
> Default I2C address: **0x27**.  Run an I2C scanner sketch if not found.

### ESP32 ↔ Passive Buzzer

```
ESP32 Pin   Buzzer
─────────   ──────
GPIO 2      (+) positive leg
GND         (−) negative leg
```
> For an **active** buzzer (beeps on its own), set `PASSIVE_BUZZER 0` in `config.h`.

### ESP32 ↔ Indicator LEDs

```
ESP32 Pin   LED         Resistor
─────────   ─────────   ──────────
GPIO 15     Green LED   220 Ω to GND
GPIO 16     Red LED     220 Ω to GND
```

### Full Schematic (ASCII)

```
                  ┌──────────────────────────────────────┐
                  │              ESP32                   │
                  │                                      │
RFID RC522        │  5 ────────────────────── SS         │
──────────────    │  18 ───────────────────── SCK        │
│ SDA ─────────── │  23 ───────────────────── MOSI       │
│ SCK ─────────── │  19 ───────────────────── MISO       │
│ MOSI ────────── │  4  ───────────────────── RST        │
│ MISO ────────── │                                      │
│ RST  ────────── │  21 ── SDA ─┐                        │
│ 3.3V ── 3.3V   │  22 ── SCL ─┤── I2C LCD 16x2        │
│ GND  ── GND    │              └── VCC ── 5V            │
──────────────    │                                      │
                  │  2  ──[Buzzer+]──GND                 │
                  │  15 ──[220Ω]──[Green LED]──GND       │
                  │  16 ──[220Ω]──[Red LED]──GND         │
                  │                                      │
                  │  [USB / 5V Power]                    │
                  └──────────────────────────────────────┘
```

---

## File Structure

```
josephite/
├── firmware/
│   └── rfid_attendance/
│       ├── rfid_attendance.ino   ← Flash this to ESP32
│       └── config.h              ← Edit WiFi/URL/timezone here
├── google-script/
│   ├── Code.gs                   ← Paste into Apps Script
│   └── Dashboard.html            ← Auto-served by Apps Script
└── RFID_ATTENDANCE_SYSTEM.md     ← This file
```

---

## Setup Guide

### Step 1 — Arduino IDE Libraries

Install these via **Sketch → Include Library → Manage Libraries**:

| Library              | Author              |
|----------------------|---------------------|
| MFRC522              | GithubCommunity     |
| LiquidCrystal_I2C    | Frank de Brabander  |
| ArduinoJson          | Benoit Blanchon     |

Board: **ESP32 Dev Module** (install via Boards Manager → `esp32` by Espressif)

---

### Step 2 — Google Sheets & Apps Script

1. Open [Google Sheets](https://sheets.google.com) → create a new sheet.
2. **Extensions → Apps Script**.
3. Delete the default `Code.gs` content; paste `google-script/Code.gs`.
4. Create a new file called **Dashboard** (HTML file) → paste `google-script/Dashboard.html`.
5. **Run → `initializeSheets()`** — authorise when prompted. This creates all sheets and headers.
6. **Run → `setupTriggers()`** — creates the 6 PM daily absent-marking trigger.
7. **Deploy → New Deployment → Web App**:
   - Execute as: **Me**
   - Who has access: **Anyone**
8. Copy the Web App URL (looks like `https://script.google.com/macros/s/ABC.../exec`).

---

### Step 3 — ESP32 Firmware

1. Open `firmware/rfid_attendance/rfid_attendance.ino` in Arduino IDE.
2. Edit **`config.h`**:
   ```cpp
   #define WIFI_SSID     "YourNetwork"
   #define WIFI_PASSWORD "YourPassword"
   #define GAS_URL       "https://script.google.com/macros/s/YOUR_ID/exec"
   #define GMT_OFFSET_SEC  28800   // UTC+8 for PH; adjust for your timezone
   ```
3. Select board: **ESP32 Dev Module**, port: your COM/ttyUSB port.
4. Flash (**Ctrl+U**).

---

### Step 4 — Register RFID Cards

1. Open the Google Sheet → **Users** tab.
2. Add rows in this format:

| UID           | Name       | Department | Email                | Role    | Date Added |
|---------------|------------|------------|----------------------|---------|------------|
| AA:BB:CC:DD   | Maria Cruz | IT         | maria@example.com    | Staff   | 2024-01-15 |

> **Tip:** Scan an unknown card once — it appears in the Records sheet with `(Unknown)`. Copy its UID from column B into the Users sheet.

---

## Smart Features

| Feature | Description |
|---------|-------------|
| **Auto toggle** | First tap → CHECK_IN; second tap → CHECK_OUT; third tap → CHECK_IN again |
| **Late detection** | After 09:00 → flagged `LATE` in records + yellow row highlight |
| **Overtime detection** | CHECK_OUT after 18:00 → flagged `OVERTIME` |
| **Early departure** | CHECK_OUT before 17:00 → flagged `EARLY_DEPARTURE` |
| **Hours worked** | Automatically calculated from CHECK_IN to CHECK_OUT |
| **Offline queue** | Up to 50 records stored in ESP32 flash; auto-synced when WiFi restores |
| **Duplicate guard** | 2-second hardware debounce + server-side timestamp dedup for offline sync |
| **Auto absent marking** | Daily trigger at 18:00 marks every user without a CHECK_IN as `ABSENT` |
| **Daily email report** | Admin receives present/late/absent summary after absent marking runs |
| **NTP time sync** | No RTC needed — synced from `pool.ntp.org` on boot |
| **Live clock on LCD** | Shows current date/time while idle |
| **Buzzer patterns** | Rising 2-tone = check-in · Falling 2-tone = check-out · Double buzz = error |
| **Row colour coding** | Green = on-time · Yellow = late · Blue = check-out · Red = absent/unknown |
| **Live dashboard** | Served from Apps Script; auto-refreshes every 30 s; 7-day trend chart |
| **CSV export** | One-click export filtered by date from the dashboard |
| **WiFi watchdog** | Checks connection every 30 s, reconnects automatically |

---

## Buzzer Sound Guide

| Event | Pattern | Meaning |
|-------|---------|---------|
| Boot | 3 short beeps | System starting |
| Check-In | Low → High (2 notes) | Welcome! |
| Check-Out | High → Low (2 notes) | Goodbye! |
| Error / Unknown | Double low buzz | Card not registered |
| Offline save | Single mid beep | Saved to local queue |

---

## Google Sheets Layout

| Sheet | Purpose |
|-------|---------|
| **Records** | Every check-in/out event, auto colour-coded |
| **Users** | Registered RFID cards ↔ names |
| **Settings** | Configurable thresholds (late hour, overtime hour, etc.) |
| **Daily Summary** | Aggregated daily counts for charting |

---

## Configurable Settings (Settings Sheet)

| Key | Default | Description |
|-----|---------|-------------|
| `late_hour` | 9 | Hour after which arrival is "Late" |
| `late_minute` | 0 | Minute threshold |
| `expected_out_hour` | 17 | Standard checkout time |
| `overtime_hour` | 18 | Overtime threshold |
| `notify_late` | FALSE | Email late employees |
| `notify_absent` | TRUE | Send daily absent email |
| `admin_email` | — | Recipient for daily report |

Changes in the Settings sheet take effect immediately on the next scan — no firmware update needed.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| LCD shows garbage | Check I2C address (try `0x3F`); verify SDA/SCL pins |
| Cards not reading | Check SPI wiring; verify MFRC522 is on 3.3 V, not 5 V |
| No WiFi | Check SSID/password in `config.h`; verify 2.4 GHz network |
| Server not responding | Redeploy Apps Script as web app; paste new URL in `config.h` |
| Time shows 1970 | NTP failed; check WiFi; increase `waitNTP()` retry count |
| Offline records not syncing | Ensure queue size < 50; check `q_size` key in Preferences |
| LCD freezes after scan | Reduce `delay()` in `showAttendanceResult()` |
