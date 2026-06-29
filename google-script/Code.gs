/*
 * ================================================================
 *   Smart Attendance System — Google Apps Script Backend  v4.0
 *   TechSei Lab
 * ================================================================
 *   First-time setup:
 *     1. Bind this script to a Google Sheet
 *     2. Run  setup_()  once (Extensions → Apps Script → Run)
 *     3. Deploy → New Deployment → Web App
 *        Execute as: Me  |  Who has access: Anyone
 *     4. Paste the Web App URL into config.h on the ESP32
 * ================================================================
 */

var SHEET_ROSTER = "Roster";
var SHEET_CTRL   = "Control";
var ATT_STUDENT  = "Student Attendance";
var ATT_STAFF    = "Staff Attendance";
var ATT_OTHER    = "Other Attendance";

function ss_() {
  return SpreadsheetApp.getActiveSpreadsheet();
}

// ════════════════════════════════════════════════════════════════
//  ENTRY POINTS
// ════════════════════════════════════════════════════════════════

function doGet(e)  { return handle(e); }
function doPost(e) { return handle(e); }

function handle(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    setup_();
    var p = (e && e.parameter) ? e.parameter : {};

    // Also accept JSON POST body
    if (e && e.postData && e.postData.contents) {
      try { var body = JSON.parse(e.postData.contents); for (var k in body) p[k] = body[k]; }
      catch (ex) {}
    }

    var action = p.action || "log";
    var out;
    switch (action) {
      case "log":           out = logAttendance_(p);  break;
      case "pushScan":      out = pushScan_(p);       break;
      case "getMode":       out = getMode_();         break;
      case "setMode":       out = setMode_(p);        break;
      case "getPending":    out = getPending_();      break;
      case "getRoster":     out = getRoster_();       break;
      case "enroll":        out = enroll_(p);         break;
      case "deleteStudent": out = deleteRec_(p);      break;
      case "getAttendance": out = getAttendance_(p);  break;
      case "stats":         out = stats_(p);          break;
      default:              out = {ok: false, msg: "unknown action"};
    }
    return json_(out, p.callback);
  } catch (err) {
    Logger.log("handle error: " + err);
    return json_({ok: false, msg: String(err)},
                 (e && e.parameter) ? e.parameter.callback : null);
  } finally {
    lock.releaseLock();
  }
}

// ════════════════════════════════════════════════════════════════
//  ONE-TIME SHEET SETUP  (run manually once)
// ════════════════════════════════════════════════════════════════

function setup_() {
  var s = ss_();
  if (s.getSheetByName(SHEET_ROSTER)) return;   // already done

  ensure_(s, SHEET_ROSTER, [
    "UID","Type","Name","Branch","ID Number","Contact",
    "Course","Batch","Instructor","Role","Department",
    "Note","Parent Phone","Photo","Active"
  ]);
  ensure_(s, ATT_STUDENT, [
    "Date","UID","Name","Branch","Course","Batch","Instructor",
    "In Time","Out Time","Hours"
  ]);
  ensure_(s, ATT_STAFF, [
    "Date","UID","Name","Branch","Role","Department",
    "In Time","Out Time","Hours"
  ]);
  ensure_(s, ATT_OTHER, [
    "Date","UID","Name","Branch","Note","In Time","Out Time","Hours"
  ]);

  var ctrl = s.getSheetByName(SHEET_CTRL);
  if (!ctrl) {
    ctrl = s.insertSheet(SHEET_CTRL);
    ctrl.appendRow(["key", "value"]);
    ctrl.appendRow(["adminMode",   "0"]);
    ctrl.appendRow(["pendingUID",  ""]);
    ctrl.appendRow(["pendingTime", ""]);
  }
  Logger.log("Sheets created.");
}

function ensure_(s, name, header) {
  var sh = s.getSheetByName(name);
  if (!sh) {
    sh = s.insertSheet(name);
    sh.appendRow(header);
    sh.getRange(1, 1, 1, header.length)
      .setBackground("#4A90D9").setFontColor("#FFFFFF").setFontWeight("bold");
    sh.setFrozenRows(1);
  }
}

// ════════════════════════════════════════════════════════════════
//  COLUMN MAPS
//  Student : Date(1)|UID(2)|Name(3)|Branch(4)|Course(5)|Batch(6)|Instructor(7)|In(8)|Out(9)|Hours(10)
//  Staff   : Date(1)|UID(2)|Name(3)|Branch(4)|Role(5)|Dept(6)|In(7)|Out(8)|Hours(9)
//  Others  : Date(1)|UID(2)|Name(3)|Branch(4)|Note(5)|In(6)|Out(7)|Hours(8)
// ════════════════════════════════════════════════════════════════

function cols_(type) {
  if (type === "Staff")  return {in: 7, out: 8, hr: 9};
  if (type === "Others") return {in: 6, out: 7, hr: 8};
  return                        {in: 8, out: 9, hr: 10};
}

function tabForType_(type) {
  if (type === "Staff")  return ATT_STAFF;
  if (type === "Others") return ATT_OTHER;
  return ATT_STUDENT;
}

// ════════════════════════════════════════════════════════════════
//  ATTENDANCE LOGGING
// ════════════════════════════════════════════════════════════════

function logAttendance_(p) {
  var uid  = String(p.uid  || "").trim();
  var date = String(p.date || "").trim();
  var inT  = String(p.intime  || "").trim();
  var outT = String(p.outtime || "").trim();

  if (!uid || !date)   return {ok: false, msg: "missing uid/date", event: ""};
  if (!inT && !outT)   return {ok: false, msg: "no time provided", event: ""};

  var rec = findRec_(uid);
  if (!rec) return {ok: false, msg: "uid not in roster: " + uid, event: ""};

  var type = (rec.type === "Staff" || rec.type === "Others") ? rec.type : "Student";
  var c    = cols_(type);
  var sh   = ss_().getSheetByName(tabForType_(type));
  var d    = sh.getDataRange().getDisplayValues();

  // Find last row for this uid on this date
  var row = -1;
  for (var i = d.length - 1; i >= 1; i--) {
    if (d[i][1] === uid && d[i][0] === date) { row = i + 1; break; }
  }

  var late = false;
  if (inT) {
    // Detect late (after 09:00)
    var parts = inT.split(":");
    if (parts.length === 2) {
      late = (parseInt(parts[0]) > 9) ||
             (parseInt(parts[0]) === 9 && parseInt(parts[1]) > 0);
    }

    if (row === -1) {
      sh.appendRow(buildRow_(type, date, uid, rec, inT, ""));
      row = sh.getLastRow();
    } else {
      var rIn  = d[row - 1][c.in - 1];
      var rOut = d[row - 1][c.out - 1];
      if (rIn && rOut) {
        // Both columns filled — new session
        sh.appendRow(buildRow_(type, date, uid, rec, inT, ""));
      } else {
        sh.getRange(row, c.in).setValue(inT);
      }
    }
    colorRow_(sh, sh.getLastRow(), late ? "#FFF3CD" : "#D4EDDA");
    return {ok: true, event: "IN",  name: rec.name, type: type, late: late};
  }

  // OUT
  if (row === -1) {
    sh.appendRow(buildRow_(type, date, uid, rec, "", outT));
    row = sh.getLastRow();
  } else {
    sh.getRange(row, c.out).setValue(outT);
    var inVal = d[row - 1][c.in - 1];
    if (inVal) sh.getRange(row, c.hr).setValue(diffHours_(inVal, outT));
  }
  colorRow_(sh, row, "#CCE5FF");
  return {ok: true, event: "OUT", name: rec.name, type: type, late: false};
}

function buildRow_(type, date, uid, rec, inT, outT) {
  if (type === "Staff")
    return [date, uid, rec.name, rec.branch, rec.role, rec.department, inT, outT, ""];
  if (type === "Others")
    return [date, uid, rec.name, rec.branch, rec.note, inT, outT, ""];
  return   [date, uid, rec.name, rec.branch, rec.course, rec.batch, rec.instructor, inT, outT, ""];
}

function colorRow_(sh, row, color) {
  var ncols = sh.getLastColumn();
  sh.getRange(row, 1, 1, ncols).setBackground(color);
}

function diffHours_(a, b) {
  try {
    var x = String(a).split(":"), y = String(b).split(":");
    var m = (parseInt(y[0]) * 60 + parseInt(y[1])) -
            (parseInt(x[0]) * 60 + parseInt(x[1]));
    return m > 0 ? (m / 60).toFixed(2) : "0.00";
  } catch (e) { return ""; }
}

// ════════════════════════════════════════════════════════════════
//  ADMIN / CONTROL SHEET
// ════════════════════════════════════════════════════════════════

function ctrlGet_(key) {
  var sh = ss_().getSheetByName(SHEET_CTRL);
  var d  = sh.getDataRange().getValues();
  for (var i = 1; i < d.length; i++) {
    if (String(d[i][0]) === key) return String(d[i][1]);
  }
  return "";
}

function ctrlSet_(key, val) {
  var sh = ss_().getSheetByName(SHEET_CTRL);
  var d  = sh.getDataRange().getValues();
  for (var i = 1; i < d.length; i++) {
    if (String(d[i][0]) === key) { sh.getRange(i + 1, 2).setValue(val); return; }
  }
  sh.appendRow([key, val]);
}

function getMode_()  { return {ok: true, adminMode: ctrlGet_("adminMode")}; }

function setMode_(p) {
  var m = (p.mode === "1") ? "1" : "0";
  ctrlSet_("adminMode", m);
  if (m === "0") ctrlSet_("pendingUID", "");
  return {ok: true, adminMode: m};
}

// Called by ESP32 when a card is scanned in admin mode
function pushScan_(p) {
  var uid = String(p.uid || "").trim();
  if (!uid) return {ok: false, msg: "no uid"};
  ctrlSet_("pendingUID",  uid);
  ctrlSet_("pendingTime", new Date().toISOString());

  // Tell the ESP32 whether this card is already enrolled
  var existing = findRec_(uid);
  return {
    ok:       true,
    msg:      "scan received",
    existing: existing !== null,
    name:     existing ? existing.name : ""
  };
}

// Called by dashboard to get the latest pending card
function getPending_() {
  var uid  = ctrlGet_("pendingUID");
  var t    = ctrlGet_("pendingTime");
  var known = uid ? findRec_(uid) : null;
  return {ok: true, uid: uid, time: t, known: known};
}

// ════════════════════════════════════════════════════════════════
//  ROSTER  (all registered cards)
// ════════════════════════════════════════════════════════════════

function rosterRows_() {
  return ss_().getSheetByName(SHEET_ROSTER).getDataRange().getValues();
}

// Map a Roster row array to an object
// Columns: UID(0)|Type(1)|Name(2)|Branch(3)|ID Number(4)|Contact(5)|
//          Course(6)|Batch(7)|Instructor(8)|Role(9)|Department(10)|
//          Note(11)|Parent Phone(12)|Photo(13)|Active(14)
function rowToRec_(r) {
  return {
    uid: String(r[0]), type: String(r[1] || "Student"),
    name: String(r[2]), branch: String(r[3]),
    idNumber: String(r[4]), contact: String(r[5]),
    course: String(r[6]), batch: String(r[7]), instructor: String(r[8]),
    role: String(r[9]), department: String(r[10]),
    note: String(r[11]), parentPhone: String(r[12]),
    photo: String(r[13]), active: String(r[14] || "1")
  };
}

function findRec_(uid) {
  var d = rosterRows_();
  for (var i = 1; i < d.length; i++) {
    if (String(d[i][0]).toLowerCase().trim() === uid.toLowerCase().trim())
      return rowToRec_(d[i]);
  }
  return null;
}

function getRoster_() {
  var d = rosterRows_(), list = [];
  for (var i = 1; i < d.length; i++) {
    if (!d[i][0]) continue;
    list.push(rowToRec_(d[i]));
  }
  return {ok: true, students: list};
}

// Enroll or update a card in the roster
function enroll_(p) {
  var uid = String(p.uid || "").trim();
  if (!uid) return {ok: false, msg: "no uid"};

  var sh  = ss_().getSheetByName(SHEET_ROSTER);
  var d   = sh.getDataRange().getValues();
  var row = -1;
  for (var i = 1; i < d.length; i++) {
    if (String(d[i][0]).toLowerCase().trim() === uid.toLowerCase()) { row = i + 1; break; }
  }

  var vals = [
    uid,
    p.type        || "Student",
    p.name        || "",
    p.branch      || "",
    p.idNumber    || "",
    p.contact     || "",
    p.course      || "",
    p.batch       || "",
    p.instructor  || "",
    p.role        || "",
    p.department  || "",
    p.note        || "",
    p.parentPhone || "",
    p.photo       || "",
    "1"
  ];

  if (row === -1) {
    sh.appendRow(vals);
  } else {
    sh.getRange(row, 1, 1, vals.length).setValues([vals]);
  }

  // Clear pending UID from control sheet
  ctrlSet_("pendingUID", "");
  return {ok: true, msg: "saved", uid: uid, name: p.name || ""};
}

function deleteRec_(p) {
  var uid = String(p.uid || "").trim();
  var sh  = ss_().getSheetByName(SHEET_ROSTER);
  var d   = sh.getDataRange().getValues();
  for (var i = 1; i < d.length; i++) {
    if (String(d[i][0]).toLowerCase().trim() === uid.toLowerCase()) {
      sh.deleteRow(i + 1);
      return {ok: true};
    }
  }
  return {ok: false, msg: "not found"};
}

// ════════════════════════════════════════════════════════════════
//  ATTENDANCE READ  (for dashboard table)
// ════════════════════════════════════════════════════════════════

function getAttendance_(p) {
  var type       = p.type || "Student";
  var filterDate = p.date || "";
  var sh         = ss_().getSheetByName(tabForType_(type));
  var d          = sh.getDataRange().getDisplayValues();
  var c          = cols_(type);
  var rows       = [];

  for (var i = 1; i < d.length; i++) {
    if (filterDate && d[i][0] !== filterDate) continue;
    var r = d[i];
    rows.push({
      date:     r[0], uid: r[1], name: r[2], branch: r[3],
      sub:      (type === "Student") ? r[4] + " / " + r[5] :
                (type === "Staff")   ? r[4] + " / " + r[5] : r[4],
      inTime:   r[c.in - 1],
      outTime:  r[c.out - 1],
      hours:    r[c.hr - 1]
    });
  }
  return {ok: true, rows: rows};
}

// ════════════════════════════════════════════════════════════════
//  STATS
// ════════════════════════════════════════════════════════════════

function stats_(p) {
  var type  = p.type  || "Student";
  var today = p.date  || fmtDate_(new Date());
  var att   = ss_().getSheetByName(tabForType_(type)).getDataRange().getDisplayValues();
  var c     = cols_(type);

  var d = rosterRows_(), roster = [];
  for (var i = 1; i < d.length; i++) {
    if (!d[i][0]) continue;
    var rr = rowToRec_(d[i]);
    if ((rr.type || "Student") === type && rr.active !== "0") roster.push(rr);
  }

  var present = {}, byDate = {}, byGroup = {};
  for (var j = 1; j < att.length; j++) {
    var date = att[j][0], uid = att[j][1];
    if (!date || !uid) continue;
    byDate[date] = (byDate[date] || 0) + 1;
    if (date === today && !present[uid]) {
      present[uid] = true;
      var g = (type === "Staff")   ? att[j][5] :
              (type === "Others")  ? att[j][3] : att[j][4];
      if (g) byGroup[g] = (byGroup[g] || 0) + 1;
    }
  }

  var absent = [];
  for (var k = 0; k < roster.length; k++) {
    if (!present[roster[k].uid])
      absent.push({name: roster[k].name, uid: roster[k].uid,
                   sub: roster[k].course || roster[k].role || ""});
  }

  var dates = Object.keys(byDate).sort().slice(-7);
  var trend = dates.map(function(dd) { return {date: dd, count: byDate[dd]}; });

  return {
    ok: true, type: type, date: today,
    total: roster.length,
    presentToday: Object.keys(present).length,
    absentToday: absent.length,
    absentees: absent, byGroup: byGroup,
    trend: trend
  };
}

// ════════════════════════════════════════════════════════════════
//  UTILITIES
// ════════════════════════════════════════════════════════════════

function fmtDate_(d) {
  function z(n) { return (n < 10 ? "0" : "") + n; }
  return z(d.getDate()) + "/" + z(d.getMonth() + 1) + "/" + d.getFullYear();
}

function json_(obj, callback) {
  var txt = JSON.stringify(obj);
  if (callback)
    return ContentService.createTextOutput(callback + "(" + txt + ");")
      .setMimeType(ContentService.MimeType.JAVASCRIPT);
  return ContentService.createTextOutput(txt)
    .setMimeType(ContentService.MimeType.JSON);
}
