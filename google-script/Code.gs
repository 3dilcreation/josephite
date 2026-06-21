/*
 * ================================================================
 *   SMART RFID ATTENDANCE SYSTEM — Google Apps Script Backend
 *
 *   Setup steps:
 *     1. Open a new Google Sheet
 *     2. Extensions → Apps Script → paste this code
 *     3. Run initializeSheets() once to create all sheets/headers
 *     4. Run setupTriggers() once to enable daily absent-marking
 *     5. Deploy → New Deployment → Web App
 *        Execute as: Me   |  Who has access: Anyone
 *     6. Copy the Web App URL into config.h on the ESP32
 *
 *   Smart features:
 *     • Auto toggle CHECK_IN ↔ CHECK_OUT per UID per day
 *     • Late / Early Departure / Overtime auto-flagging
 *     • Hours-worked calculation per session
 *     • Offline-synced records merged without duplicates
 *     • Daily absent auto-marking via time trigger (6 PM)
 *     • Daily summary email to admin
 *     • Late-arrival notification to employee (optional)
 *     • Row colour-coding in Google Sheets
 *     • Live HTML dashboard (doGet serves it)
 * ================================================================
 */

// ── Configuration ─────────────────────────────────────────────
const CONFIG = {
  adminEmail:    'admin@yourschool.com',
  timezone:      'Asia/Manila',    // Change to your timezone
  lateHour:      9,                // After 09:00 = Late
  lateMinute:    0,
  expectedOutHr: 17,               // Expected checkout hour
  overtimeHr:    18,               // After 18:00 = Overtime
  notifyLate:    false,            // Email employees on late arrival
  notifyAbsent:  true,             // Email admin daily absent report
};

const SHEETS = {
  records:  'Records',
  users:    'Users',
  settings: 'Settings',
  summary:  'Daily Summary',
};

// ══════════════════════════════════════════════════════════════
//  WEB APP ENDPOINTS
// ══════════════════════════════════════════════════════════════

function doPost(e) {
  try {
    if (!e || !e.postData) throw new Error('No POST body received');
    const data   = JSON.parse(e.postData.contents);
    const result = processAttendance(data);
    return jsonResponse(result);
  } catch (err) {
    Logger.log('doPost error: ' + err);
    return jsonResponse({ status: 'error', message: err.message });
  }
}

function doGet(e) {
  const action = (e && e.parameter && e.parameter.action) || '';

  switch (action) {
    case 'getData':   return getAttendanceData(e.parameter);
    case 'getStats':  return getTodayStats();
    case 'getUsers':  return getUserList();
    case 'export':    return exportCSV(e.parameter);
    default:
      return HtmlService
        .createTemplateFromFile('Dashboard')
        .evaluate()
        .setTitle('Smart Attendance Dashboard')
        .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
  }
}

// ══════════════════════════════════════════════════════════════
//  CORE ATTENDANCE LOGIC
// ══════════════════════════════════════════════════════════════

function processAttendance(data) {
  const ss      = getSpreadsheet();
  const recSh   = ss.getSheetByName(SHEETS.records);
  const usersSh = ss.getSheetByName(SHEETS.users);

  if (!recSh || !usersSh) {
    initializeSheets();
    return { status: 'error', message: 'Sheets created. Retry in a moment.' };
  }

  const uid        = String(data.uid || '').toUpperCase().trim();
  const rawTs      = data.timestamp || new Date().toISOString();
  const device     = data.device     || 'Unknown';
  const deviceId   = data.device_id  || '';
  const isSync     = !!data.offline_sync;
  const scanTime   = new Date(rawTs);

  if (!uid) return { status: 'error', message: 'Missing UID' };

  // ── Look up user ──────────────────────────────────────────
  const users    = usersSh.getDataRange().getValues();
  let userName   = null, userDept = null, userEmail = null;

  for (let i = 1; i < users.length; i++) {
    if (String(users[i][0]).toUpperCase().trim() === uid) {
      userName  = users[i][1];
      userDept  = users[i][2];
      userEmail = users[i][3];
      break;
    }
  }

  if (!userName) {
    // Log unknown card for easy registration later
    recSh.appendRow([
      scanTime, uid, '(Unknown)', '', 'UNKNOWN', 'N/A',
      device, deviceId, isSync ? 'OFFLINE_SYNC' : 'LIVE', '', '',
    ]);
    formatRow(recSh, recSh.getLastRow(), 'UNKNOWN', false);
    return { status: 'unknown', uid: uid, message: 'Card not registered' };
  }

  // ── Duplicate guard (offline sync: skip if already exists) ─
  if (isSync && recordExists(uid, scanTime, recSh)) {
    return { status: 'duplicate', message: 'Already synced' };
  }

  // ── Determine action ──────────────────────────────────────
  const action   = determineAction(uid, scanTime, recSh);
  const settings = loadSettings(ss);

  const lHour  = Number(settings.late_hour   || CONFIG.lateHour);
  const lMin   = Number(settings.late_minute || CONFIG.lateMinute);
  const outHr  = Number(settings.expected_out_hour || CONFIG.expectedOutHr);
  const otHr   = Number(settings.overtime_hour     || CONFIG.overtimeHr);

  const h = scanTime.getHours();
  const m = scanTime.getMinutes();

  const isLate          = action === 'CHECK_IN'  && (h > lHour || (h === lHour && m >= lMin));
  const isEarlyDep      = action === 'CHECK_OUT' && h < outHr;
  const isOvertime      = action === 'CHECK_OUT' && h >= otHr;

  let statusNote = '';
  if (action === 'CHECK_IN')  statusNote = isLate    ? 'LATE'            : 'ON_TIME';
  if (action === 'CHECK_OUT') statusNote = isOvertime ? 'OVERTIME'
                                         : isEarlyDep ? 'EARLY_DEPARTURE' : 'ON_TIME';

  // ── Hours worked ──────────────────────────────────────────
  let hoursWorked = '';
  if (action === 'CHECK_OUT') {
    const lastIn = getLastCheckIn(uid, recSh);
    if (lastIn) {
      hoursWorked = ((scanTime - lastIn) / 3600000).toFixed(2);
    }
  }

  // ── Append record ─────────────────────────────────────────
  recSh.appendRow([
    scanTime,                        // A Timestamp
    uid,                             // B UID
    userName,                        // C Name
    userDept,                        // D Department
    action,                          // E Action
    isLate ? 'YES' : 'NO',          // F Late
    device,                          // G Device
    deviceId,                        // H Device ID
    isSync ? 'OFFLINE_SYNC' : 'LIVE',// I Sync Type
    statusNote,                      // J Status Note
    hoursWorked,                     // K Hours Worked
  ]);
  formatRow(recSh, recSh.getLastRow(), action, isLate);

  // ── Update daily summary ──────────────────────────────────
  updateDailySummary(ss, scanTime, action, isLate);

  // ── Optional: email late employee ─────────────────────────
  if (isLate && CONFIG.notifyLate && userEmail) {
    notifyLateEmployee(userName, userEmail, scanTime);
  }

  return {
    status:     'success',
    name:       userName,
    department: userDept,
    action:     action,
    late:       isLate,
    overtime:   isOvertime,
    earlyDep:   isEarlyDep,
    statusNote: statusNote,
    hoursWorked: hoursWorked,
    message:    buildMessage(userName, action, isLate, isOvertime),
  };
}

// ── Action toggle: CHECK_IN → CHECK_OUT → CHECK_IN ... ────────
function determineAction(uid, scanTime, recSh) {
  const dateStr = fmtDate(scanTime);
  const rows    = recSh.getDataRange().getValues();

  for (let i = rows.length - 1; i >= 1; i--) {
    if (String(rows[i][1]).toUpperCase().trim() !== uid) continue;
    if (fmtDate(new Date(rows[i][0])) !== dateStr)       continue;
    const prev = rows[i][4];
    if (prev === 'CHECK_IN')  return 'CHECK_OUT';
    if (prev === 'CHECK_OUT') return 'CHECK_IN';  // 3rd tap re-enters
    break;
  }
  return 'CHECK_IN';
}

function getLastCheckIn(uid, recSh) {
  const rows = recSh.getDataRange().getValues();
  for (let i = rows.length - 1; i >= 1; i--) {
    if (String(rows[i][1]).toUpperCase().trim() === uid && rows[i][4] === 'CHECK_IN') {
      return new Date(rows[i][0]);
    }
  }
  return null;
}

function recordExists(uid, scanTime, recSh) {
  const ts   = scanTime.getTime();
  const rows = recSh.getDataRange().getValues();
  for (let i = 1; i < rows.length; i++) {
    if (String(rows[i][1]).toUpperCase().trim() !== uid) continue;
    if (Math.abs(new Date(rows[i][0]).getTime() - ts) < 5000) return true;
  }
  return false;
}

// ══════════════════════════════════════════════════════════════
//  ROW FORMATTING
// ══════════════════════════════════════════════════════════════

function formatRow(sh, row, action, late) {
  const r = sh.getRange(row, 1, 1, 11);
  const colours = {
    'CHECK_IN':  late ? '#FFF3CD' : '#D4EDDA',
    'CHECK_OUT': '#CCE5FF',
    'UNKNOWN':   '#F8D7DA',
    'ABSENT':    '#F5C6CB',
  };
  r.setBackground(colours[action] || '#FFFFFF');
}

// ══════════════════════════════════════════════════════════════
//  DAILY SUMMARY SHEET
// ══════════════════════════════════════════════════════════════

function updateDailySummary(ss, date, action, late) {
  const sh      = ss.getSheetByName(SHEETS.summary);
  if (!sh) return;

  const dateStr = fmtDate(date);
  const rows    = sh.getDataRange().getValues();
  let   rowIdx  = -1;

  for (let i = 1; i < rows.length; i++) {
    if (fmtDate(new Date(rows[i][0])) === dateStr) { rowIdx = i + 1; break; }
  }

  if (rowIdx === -1) {
    sh.appendRow([new Date(dateStr + 'T00:00:00'), 0, 0, 0, 0]);
    rowIdx = sh.getLastRow();
  }

  if (action === 'CHECK_IN') {
    const pCell = sh.getRange(rowIdx, 2);
    pCell.setValue(pCell.getValue() + 1);
    if (late) {
      const lCell = sh.getRange(rowIdx, 3);
      lCell.setValue(lCell.getValue() + 1);
    }
  }
}

// ══════════════════════════════════════════════════════════════
//  ABSENT AUTO-MARKING  (run via daily trigger at 18:00)
// ══════════════════════════════════════════════════════════════

function markAbsentees() {
  const ss      = getSpreadsheet();
  const usersSh = ss.getSheetByName(SHEETS.users);
  const recSh   = ss.getSheetByName(SHEETS.records);
  const sumSh   = ss.getSheetByName(SHEETS.summary);

  const today   = fmtDate(new Date());
  const records = recSh.getDataRange().getValues();
  const checkedInToday = new Set();

  for (let i = 1; i < records.length; i++) {
    if (fmtDate(new Date(records[i][0])) === today && records[i][4] === 'CHECK_IN') {
      checkedInToday.add(String(records[i][1]).toUpperCase().trim());
    }
  }

  const users = usersSh.getDataRange().getValues();
  let   absentCnt = 0;

  for (let i = 1; i < users.length; i++) {
    const uid = String(users[i][0]).toUpperCase().trim();
    if (!uid || checkedInToday.has(uid)) continue;

    recSh.appendRow([
      new Date(today + 'T18:00:00'),
      uid, users[i][1], users[i][2],
      'ABSENT', 'N/A', 'SYSTEM', 'SYSTEM', 'AUTO', 'ABSENT', '',
    ]);
    formatRow(recSh, recSh.getLastRow(), 'ABSENT', false);
    absentCnt++;
  }

  // Patch absent count in summary
  const sumData = sumSh.getDataRange().getValues();
  for (let i = 1; i < sumData.length; i++) {
    if (fmtDate(new Date(sumData[i][0])) === today) {
      sumSh.getRange(i + 1, 4).setValue(absentCnt);
      break;
    }
  }

  if (CONFIG.notifyAbsent) {
    sendDailyReport(new Date(), users.length - 1, checkedInToday.size, absentCnt);
  }

  Logger.log(`Absent marking done. ${absentCnt} marked absent.`);
}

// ══════════════════════════════════════════════════════════════
//  EMAIL NOTIFICATIONS
// ══════════════════════════════════════════════════════════════

function sendDailyReport(date, total, present, absent) {
  const tz      = CONFIG.timezone;
  const dateStr = Utilities.formatDate(date, tz, 'MMMM dd, yyyy');
  const rate    = total ? Math.round(present / total * 100) : 0;

  const html = `
<h2>Daily Attendance Report — ${dateStr}</h2>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
  <tr><td><b>Total Users</b></td><td>${total}</td></tr>
  <tr><td><b>Present</b></td><td style="color:green">${present}</td></tr>
  <tr><td><b>Absent</b></td><td style="color:red">${absent}</td></tr>
  <tr><td><b>Attendance Rate</b></td><td>${rate}%</td></tr>
</table>
<br>
<a href="${ScriptApp.getService().getUrl()}">Open Live Dashboard</a>
<br><br><small>Smart Attendance System — Auto-generated</small>`;

  MailApp.sendEmail({
    to:       CONFIG.adminEmail,
    subject:  `Attendance Report — ${dateStr}`,
    htmlBody: html,
  });
}

function notifyLateEmployee(name, email, time) {
  const t = Utilities.formatDate(time, CONFIG.timezone, 'hh:mm a');
  MailApp.sendEmail({
    to:      email,
    subject: 'Late Arrival Notice',
    body:    `Dear ${name},\n\nYou checked in late at ${t}.\n\nBest regards,\nAttendance System`,
  });
}

// ══════════════════════════════════════════════════════════════
//  DASHBOARD DATA APIs
// ══════════════════════════════════════════════════════════════

function getAttendanceData(params) {
  const ss     = getSpreadsheet();
  const sh     = ss.getSheetByName(SHEETS.records);
  const rows   = sh.getDataRange().getValues();
  const tz     = CONFIG.timezone;

  const fDate  = params.date  || '';
  const fName  = (params.name || '').toLowerCase();
  const fDept  = (params.dept || '').toLowerCase();
  const limit  = Math.min(Number(params.limit || 500), 1000);

  const out = [];
  for (let i = rows.length - 1; i >= 1 && out.length < limit; i--) {
    const row    = rows[i];
    const rowDt  = row[0] ? Utilities.formatDate(new Date(row[0]), tz, 'yyyy-MM-dd') : '';
    const rowNm  = String(row[2]).toLowerCase();
    const rowDep = String(row[3]).toLowerCase();

    if (fDate && rowDt !== fDate) continue;
    if (fName && !rowNm.includes(fName)) continue;
    if (fDept && !rowDep.includes(fDept)) continue;

    out.push({
      timestamp:   row[0] ? Utilities.formatDate(new Date(row[0]), tz, 'yyyy-MM-dd HH:mm:ss') : '',
      uid:         row[1],
      name:        row[2],
      department:  row[3],
      action:      row[4],
      late:        row[5],
      device:      row[6],
      deviceId:    row[7],
      syncType:    row[8],
      statusNote:  row[9],
      hoursWorked: row[10],
    });
  }

  return jsonResponse({ status: 'success', count: out.length, data: out });
}

function getTodayStats() {
  const ss     = getSpreadsheet();
  const sh     = ss.getSheetByName(SHEETS.records);
  const rows   = sh.getDataRange().getValues();
  const tz     = CONFIG.timezone;
  const today  = Utilities.formatDate(new Date(), tz, 'yyyy-MM-dd');

  let present = 0, late = 0, absent = 0, checkOuts = 0;
  const recent = [];
  const seenIn = new Set();

  for (let i = rows.length - 1; i >= 1; i--) {
    const row = rows[i];
    if (!row[0]) continue;
    const dt = Utilities.formatDate(new Date(row[0]), tz, 'yyyy-MM-dd');
    if (dt !== today) continue;

    const action = row[4];
    if (action === 'CHECK_IN') {
      if (!seenIn.has(row[1])) { seenIn.add(row[1]); present++; }
      if (row[5] === 'YES') late++;
    } else if (action === 'CHECK_OUT') checkOuts++;
    else if (action === 'ABSENT') absent++;

    if (recent.length < 15) {
      recent.push({
        time:   Utilities.formatDate(new Date(row[0]), tz, 'HH:mm'),
        name:   row[2],
        action: action,
        late:   row[5],
        dept:   row[3],
      });
    }
  }

  // Weekly trend (last 7 days)
  const trend = getWeeklyTrend(ss, tz);

  return jsonResponse({
    status: 'success',
    today, present, late, absent, checkOuts,
    recentActivity: recent,
    weeklyTrend: trend,
  });
}

function getWeeklyTrend(ss, tz) {
  const sh  = ss.getSheetByName(SHEETS.summary);
  if (!sh)  return [];
  const rows = sh.getDataRange().getValues();
  const out  = [];
  for (let i = Math.max(1, rows.length - 7); i < rows.length; i++) {
    out.push({
      date:    rows[i][0] ? Utilities.formatDate(new Date(rows[i][0]), tz, 'MM/dd') : '',
      present: rows[i][1] || 0,
      late:    rows[i][2] || 0,
      absent:  rows[i][3] || 0,
    });
  }
  return out;
}

function getUserList() {
  const ss   = getSpreadsheet();
  const sh   = ss.getSheetByName(SHEETS.users);
  const rows = sh.getDataRange().getValues();
  const out  = [];
  for (let i = 1; i < rows.length; i++) {
    if (!rows[i][0]) continue;
    out.push({ uid: rows[i][0], name: rows[i][1], department: rows[i][2] });
  }
  return jsonResponse({ status: 'success', users: out });
}

function exportCSV(params) {
  const ss   = getSpreadsheet();
  const sh   = ss.getSheetByName(SHEETS.records);
  const rows = sh.getDataRange().getValues();
  const tz   = CONFIG.timezone;
  const fDt  = params.date || '';

  const lines = [rows[0].join(',')];
  for (let i = 1; i < rows.length; i++) {
    const dt = rows[i][0] ? Utilities.formatDate(new Date(rows[i][0]), tz, 'yyyy-MM-dd') : '';
    if (fDt && dt !== fDt) continue;
    lines.push(rows[i].map(v => `"${String(v).replace(/"/g, '""')}"`).join(','));
  }

  return ContentService
    .createTextOutput(lines.join('\n'))
    .setMimeType(ContentService.MimeType.CSV);
}

// ══════════════════════════════════════════════════════════════
//  SETTINGS SHEET
// ══════════════════════════════════════════════════════════════

function loadSettings(ss) {
  const sh = ss.getSheetByName(SHEETS.settings);
  if (!sh) return {};
  const rows = sh.getDataRange().getValues();
  const out  = {};
  for (let i = 1; i < rows.length; i++) {
    if (rows[i][0]) out[rows[i][0]] = rows[i][1];
  }
  return out;
}

// ══════════════════════════════════════════════════════════════
//  ONE-TIME INITIALIZATION  (run manually once)
// ══════════════════════════════════════════════════════════════

function initializeSheets() {
  const ss = getSpreadsheet();

  // Records
  ensureSheet(ss, SHEETS.records,
    ['Timestamp','UID','Name','Department','Action','Late',
     'Device','Device ID','Sync Type','Status Note','Hours Worked'],
    '#4A90D9');

  // Users
  ensureSheet(ss, SHEETS.users,
    ['UID','Name','Department','Email','Role','Date Added'],
    '#27AE60');
  const uSh = ss.getSheetByName(SHEETS.users);
  if (uSh.getLastRow() === 1) {
    uSh.appendRow(['AA:BB:CC:DD', 'Sample User', 'IT Dept', 'user@example.com', 'Staff', new Date()]);
  }

  // Settings
  ensureSheet(ss, SHEETS.settings, ['Key','Value','Description'], '#8E44AD');
  const sSh = ss.getSheetByName(SHEETS.settings);
  if (sSh.getLastRow() === 1) {
    const defaults = [
      ['late_hour',          9,      'Hour after which check-in is Late'],
      ['late_minute',        0,      'Minute threshold for late check-in'],
      ['expected_out_hour', 17,      'Standard checkout hour'],
      ['overtime_hour',     18,      'Overtime threshold hour'],
      ['notify_late',    'FALSE',    'Email late employees (TRUE/FALSE)'],
      ['notify_absent',  'TRUE',     'Send daily absent report (TRUE/FALSE)'],
      ['admin_email', CONFIG.adminEmail, 'Admin email for reports'],
    ];
    defaults.forEach(r => sSh.appendRow(r));
  }

  // Daily Summary
  ensureSheet(ss, SHEETS.summary,
    ['Date','Present','Late','Absent','Attendance Rate'],
    '#E67E22');

  Logger.log('Sheets initialized ✓');
  SpreadsheetApp.getUi().alert('Sheets initialized successfully!');
}

function ensureSheet(ss, name, headers, colour) {
  let sh = ss.getSheetByName(name);
  if (!sh) sh = ss.insertSheet(name);
  if (sh.getLastRow() === 0) {
    sh.appendRow(headers);
    const hdr = sh.getRange(1, 1, 1, headers.length);
    hdr.setBackground(colour).setFontColor('#FFFFFF').setFontWeight('bold');
    sh.setFrozenRows(1);
    sh.setColumnWidth(1, 160);
  }
  return sh;
}

// ── Daily trigger setup (run once) ────────────────────────────
function setupTriggers() {
  ScriptApp.getProjectTriggers().forEach(t => ScriptApp.deleteTrigger(t));

  ScriptApp.newTrigger('markAbsentees')
    .timeBased()
    .everyDays(1)
    .atHour(18)
    .inTimezone(CONFIG.timezone)
    .create();

  Logger.log('Trigger created: markAbsentees @ 18:00 daily');
  SpreadsheetApp.getUi().alert('Daily trigger set up at 18:00!');
}

// ══════════════════════════════════════════════════════════════
//  UTILITIES
// ══════════════════════════════════════════════════════════════

function getSpreadsheet() {
  try { return SpreadsheetApp.getActiveSpreadsheet(); }
  catch (_) { throw new Error('Script must be bound to a Google Sheet'); }
}

function jsonResponse(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function fmtDate(d) {
  return Utilities.formatDate(d, CONFIG.timezone, 'yyyy-MM-dd');
}

function buildMessage(name, action, late, overtime) {
  if (action === 'CHECK_IN')  return late ? `Welcome ${name}! (Late)` : `Welcome ${name}!`;
  if (action === 'CHECK_OUT') return overtime ? `Goodbye ${name}! (Overtime)` : `Goodbye ${name}!`;
  return name;
}
