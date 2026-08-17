(function () {
  const payload = window.REPAIR_REVIEW_INDEX || { stats: {}, records: [] };
  const records = payload.records || [];
  const stats = payload.stats || {};
  const storageKey = "baseball_repaired_review_20260817_v1";
  const reviewState = readState();
  let filtered = records.slice();
  let currentIndex = 0;

  const gateLabels = {
    schema_gate: "结构门",
    media_gate: "媒体存在",
    audio_candidate_gate: "音频候选",
    video_contact_gate: "视频触球语义",
    audio_video_binding_gate: "音视频绑定",
    independent_review_gate: "独立复核",
    label_consistency_gate: "标签一致",
    source_traceability_gate: "来源追踪",
    media_readable_gate: "媒体可读",
    duration_gate: "时长一致",
    event_time_gate: "击球时间",
    contact_audio_gate: "击球声"
  };

  function $(selector, root = document) {
    return root.querySelector(selector);
  }

  function $all(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
  }

  function readState() {
    try {
      return JSON.parse(localStorage.getItem(storageKey) || "{}");
    } catch (error) {
      return {};
    }
  }

  function writeState() {
    localStorage.setItem(storageKey, JSON.stringify(reviewState));
  }

  function stateFor(record) {
    if (!reviewState[record.sampleId]) {
      reviewState[record.sampleId] = {
        decision: "",
        liveContact: false,
        audioTransient: false,
        labelOk: false,
        sourceTrace: false,
        notes: "",
        updatedAt: ""
      };
    }
    return reviewState[record.sampleId];
  }

  function fmt(value, suffix = "") {
    if (value === null || value === undefined || value === "") return "NA";
    return `${value}${suffix}`;
  }

  function passValue(value) {
    return value === "yes" || value === "pass" || String(value || "").startsWith("pass_");
  }

  function decisionText(decision) {
    if (decision === "correct") return "已确认正确";
    if (decision === "needs_followup") return "需复查";
    return "未标记";
  }

  function decisionBadgeClass(decision) {
    if (decision === "correct") return "good";
    if (decision === "needs_followup") return "warn";
    return "";
  }

  function uniqueValues(field) {
    return Array.from(new Set(records.map((record) => record[field]).filter(Boolean))).sort();
  }

  function initFilters() {
    const labelFilter = $("#labelFilter");
    const batchFilter = $("#batchFilter");
    labelFilter.innerHTML = '<option value="">全部标签</option>' +
      uniqueValues("label").map((value) => `<option value="${escapeAttr(value)}">${escapeHtml(value)}</option>`).join("");
    batchFilter.innerHTML = '<option value="">全部批次</option>' +
      uniqueValues("batchName").map((value) => `<option value="${escapeAttr(value)}">${escapeHtml(value)}</option>`).join("");
    ["searchInput", "labelFilter", "batchFilter", "gateFilter", "decisionFilter"].forEach((id) => {
      $("#" + id).addEventListener("input", applyFilters);
      $("#" + id).addEventListener("change", applyFilters);
    });
  }

  function renderStats() {
    $("#buildSummary").textContent =
      `${fmt(stats.generatedAt)} 生成；媒体只在本地 staged 目录存在时可播放，GitHub 分支仅保存检索页和索引。`;
    const localCounts = records.reduce((acc, record) => {
      const decision = stateFor(record).decision || "blank";
      acc[decision] = (acc[decision] || 0) + 1;
      return acc;
    }, {});
    const items = [
      ["候选已验证", stats.newlyVerifiedUniqueSampleIds, "NEWLY_VERIFIED 唯一样本"],
      ["可物化", stats.materializationReadyRows, "READY_TO_MATERIALIZE"],
      ["阻塞", stats.blockedMaterializationRows, "BLOCKED_MATERIALIZATION"],
      ["正式可直接训练", stats.formalDirectTrainableRows, "当前 checkout 仍按 0 计算"],
      ["checkout 五文件齐", stats.currentCheckoutCompleteFiveFiles, "但最终时间匹配为 0"],
      ["本地已确认", localCounts.correct || 0, "浏览器复核记录"]
    ];
    $("#statsStrip").innerHTML = items.map(([label, value, hint]) => (
      `<div class="stat"><strong>${escapeHtml(String(value ?? "NA"))}</strong><span>${escapeHtml(label)}</span><span>${escapeHtml(hint)}</span></div>`
    )).join("");
  }

  function searchableText(record) {
    return [
      record.sampleId,
      record.label,
      record.collector,
      record.batchName,
      record.batchIndex,
      record.reauditIndex,
      record.mainRelativePath,
      record.source.videoTitle,
      record.source.videoUrl,
      record.source.sourceId,
      record.source.clipId,
      record.paths.materializedRelativePath
    ].join(" ").toLowerCase();
  }

  function applyFilters() {
    const query = $("#searchInput").value.trim().toLowerCase();
    const label = $("#labelFilter").value;
    const batch = $("#batchFilter").value;
    const gate = $("#gateFilter").value;
    const decision = $("#decisionFilter").value;

    filtered = records.filter((record) => {
      const state = stateFor(record);
      const actualDecision = state.decision || "blank";
      const ready = record.readiness.materializationReady === "yes" &&
        record.readiness.allListedGatesPass &&
        record.readiness.stagedFiveFilesComplete;
      return (!query || searchableText(record).includes(query)) &&
        (!label || record.label === label) &&
        (!batch || record.batchName === batch) &&
        (!gate || (gate === "ready" ? ready : !ready)) &&
        (!decision || actualDecision === decision);
    });

    if (!filtered.includes(filtered[currentIndex])) currentIndex = 0;
    renderList();
    renderDetail();
  }

  function renderList() {
    $("#resultCount").textContent = `显示 ${filtered.length} / ${records.length} 条`;
    const active = filtered[currentIndex];
    $("#sampleList").innerHTML = filtered.map((record, index) => {
      const state = stateFor(record);
      const ready = record.readiness.materializationReady === "yes" &&
        record.readiness.allListedGatesPass &&
        record.readiness.stagedFiveFilesComplete;
      return `
        <div class="sample-item ${active && active.sampleId === record.sampleId ? "active" : ""}" data-index="${index}">
          <div class="sample-title">
            <strong>${escapeHtml(record.order + ". " + record.sampleId)}</strong>
            <span class="source-title">${escapeHtml(record.source.videoTitle || record.mainRelativePath)}</span>
          </div>
          <div class="sample-meta">
            <div>${escapeHtml(record.label)}</div>
            <span class="badge ${ready ? "good" : "bad"}">${ready ? "ready" : "check"}</span>
            <span class="badge ${decisionBadgeClass(state.decision)}">${decisionText(state.decision)}</span>
          </div>
        </div>`;
    }).join("");
    $all(".sample-item").forEach((item) => {
      item.addEventListener("click", () => {
        currentIndex = Number(item.dataset.index || 0);
        renderList();
        renderDetail();
      });
    });
  }

  function renderDetail() {
    const record = filtered[currentIndex];
    const detail = $("#sampleDetail");
    if (!record) {
      detail.innerHTML = '<div class="empty-state">没有匹配样本。</div>';
      return;
    }

    const template = $("#detailTemplate").content.cloneNode(true);
    const state = stateFor(record);
    const ready = record.readiness.materializationReady === "yes" &&
      record.readiness.allListedGatesPass &&
      record.readiness.stagedFiveFilesComplete;

    $('[data-field="batch"]', template).textContent =
      `${record.batchName} #${record.batchIndex} · reaudit ${record.reauditIndex}`;
    $('[data-field="sampleId"]', template).textContent = record.sampleId;
    $('[data-field="headBadges"]', template).innerHTML = [
      badge(record.label),
      badge(ready ? "可物化" : "需注意", ready ? "good" : "bad"),
      badge(`正式直接训练: ${stats.formalDirectTrainableRows || 0}`, "warn"),
      badge(decisionText(state.decision), decisionBadgeClass(state.decision))
    ].join("");

    $('[data-field="videoClock"]', template).textContent =
      `final ${fmt(record.timing.finalEventStart, "s")} - ${fmt(record.timing.finalEventEnd, "s")}`;
    $('[data-field="audioClock"]', template).textContent =
      `peak ${fmt(record.timing.materializedAudioPeakTime, "s")}`;
    $('[data-field="timingList"]', template).innerHTML = dlRows([
      ["旧时间", `${fmt(record.timing.eventStartBefore, "s")} - ${fmt(record.timing.eventEndBefore, "s")}`],
      ["最终时间", `${fmt(record.timing.finalEventStart, "s")} - ${fmt(record.timing.finalEventEnd, "s")}`],
      ["视频触球", fmt(record.timing.visualContactTime, "s")],
      ["音频候选", fmt(record.timing.audioCandidateTime, "s")],
      ["物化峰值", fmt(record.timing.materializedAudioPeakTime, "s")],
      ["音频指标", record.timing.materializedAudioMetrics || "NA"],
      ["视频/音频时长", `${fmt(record.timing.videoDurationSec, "s")} / ${fmt(record.timing.audioDurationSec, "s")}`]
    ]);
    $('[data-field="sampleInfo"]', template).innerHTML = dlRows([
      ["sample_id", record.sampleId],
      ["label", record.label],
      ["collector", record.collector],
      ["main_relative_path", record.mainRelativePath],
      ["staged 路径", record.paths.materializedRelativePath],
      ["sample.csv 时间", `${fmt(record.sampleCsv.event_start, "s")} - ${fmt(record.sampleCsv.event_end, "s")}`],
      ["landing_zone", record.sampleCsv.landing_zone || "pending"],
      ["trajectory_type", record.sampleCsv.trajectory_type || "NA"],
      ["五文件", fiveFileSummary(record.files)],
      ["checkout 状态", `${record.readiness.currentCheckoutCompleteFiveFiles} complete / ${record.readiness.currentCheckoutTimeMatchesFinal} final-time-match`]
    ]);
    $('[data-field="gateList"]', template).innerHTML = Object.entries(record.gates).map(([name, value]) => {
      const ok = passValue(value);
      return `<div class="gate ${ok ? "pass" : "fail"}"><span class="gate-name">${escapeHtml(gateLabels[name] || name)}</span><strong>${escapeHtml(value || "blank")}</strong></div>`;
    }).join("");
    $('[data-field="sourceInfo"]', template).innerHTML = dlRows([
      ["video_title", record.source.videoTitle || "NA"],
      ["video_url", sourceLink(record.source.videoUrl)],
      ["source_id", record.source.sourceId || "NA"],
      ["clip_id", record.source.clipId || "NA"],
      ["source_path", record.source.sourcePath || "NA"],
      ["源片段时间", `${fmt(record.source.clipStartTime, "s")} - ${fmt(record.source.clipEndTime, "s")}`],
      ["审核视频路径", record.paths.reviewMediaVideoPath || "NA"],
      ["审核音频路径", record.paths.reviewMediaAudioPath || "NA"],
      ["审核输入", record.paths.reviewMediaSourceFile || "NA"],
      ["证据目录", record.paths.evidencePath || "NA"],
      ["复核输出", record.paths.reviewOutputPath || "NA"],
      ["sample.csv", `<a href="${escapeAttr(record.paths.sampleCsvUrl)}" target="_blank">打开</a>`],
      ["source.txt", `<a href="${escapeAttr(record.paths.sourceTxtUrl)}" target="_blank">打开</a>`]
    ], true);

    const video = $("#sampleVideo", template);
    const audio = $("#sampleAudio", template);
    video.src = record.paths.videoUrl;
    audio.src = record.paths.audioUrl;

    $("#localStatus", template).textContent = decisionText(state.decision);
    $all("[data-check]", template).forEach((input) => {
      input.checked = Boolean(state[input.dataset.check]);
      input.addEventListener("change", () => {
        state[input.dataset.check] = input.checked;
        state.updatedAt = new Date().toISOString();
        writeState();
        renderStats();
        renderList();
      });
    });
    $("#reviewNotes", template).value = state.notes || "";
    $("#reviewNotes", template).addEventListener("input", (event) => {
      state.notes = event.target.value;
      state.updatedAt = new Date().toISOString();
      writeState();
    });
    $("#markCorrectBtn", template).addEventListener("click", () => setDecision(record, "correct"));
    $("#markFollowupBtn", template).addEventListener("click", () => setDecision(record, "needs_followup"));
    $("#clearDecisionBtn", template).addEventListener("click", () => setDecision(record, ""));
    $("#seekFinalBtn", template).addEventListener("click", () => seekMedia(record, video, audio, 0));
    $("#backBtn", template).addEventListener("click", () => seekMedia(record, video, audio, -0.05));
    $("#forwardBtn", template).addEventListener("click", () => seekMedia(record, video, audio, 0.05));
    $("#copyPathBtn", template).addEventListener("click", () => copyText(record.paths.materializedRelativePath));

    detail.innerHTML = "";
    detail.appendChild(template);
  }

  function setDecision(record, decision) {
    const state = stateFor(record);
    state.decision = decision;
    state.updatedAt = new Date().toISOString();
    writeState();
    renderStats();
    renderList();
    renderDetail();
  }

  function seekMedia(record, video, audio, offset) {
    const center = Number(record.timing.finalCenter ?? record.timing.materializedAudioPeakTime ?? 0) + offset;
    if (Number.isFinite(center)) {
      video.currentTime = Math.max(0, center);
      audio.currentTime = Math.max(0, center);
    }
  }

  function sourceLink(url) {
    if (!url) return "NA";
    return `<a href="${escapeAttr(url)}" target="_blank" rel="noreferrer">${escapeHtml(url)}</a>`;
  }

  function fiveFileSummary(files) {
    return Object.entries(files).map(([name, info]) => {
      const ok = info.exists && info.sizeBytes > 0;
      return `${name}:${ok ? "yes" : "no"}`;
    }).join(" / ");
  }

  function dlRows(rows, trustedHtml = false) {
    return rows.map(([label, value]) => (
      `<dt>${escapeHtml(label)}</dt><dd>${trustedHtml ? value : escapeHtml(String(value ?? ""))}</dd>`
    )).join("");
  }

  function badge(text, type = "") {
    return `<span class="badge ${escapeAttr(type)}">${escapeHtml(text)}</span>`;
  }

  function copyText(text) {
    navigator.clipboard?.writeText(text).catch(() => {});
  }

  function exportRows() {
    return records.map((record) => {
      const state = stateFor(record);
      return {
        sample_id: record.sampleId,
        label: record.label,
        batch_name: record.batchName,
        batch_index: record.batchIndex,
        final_event_start: record.timing.finalEventStart,
        final_event_end: record.timing.finalEventEnd,
        materialized_relative_path: record.paths.materializedRelativePath,
        decision: state.decision || "",
        live_contact_checked: state.liveContact ? "yes" : "no",
        audio_transient_checked: state.audioTransient ? "yes" : "no",
        label_checked: state.labelOk ? "yes" : "no",
        source_trace_checked: state.sourceTrace ? "yes" : "no",
        notes: state.notes || "",
        updated_at: state.updatedAt || ""
      };
    });
  }

  function download(name, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function toCsv(rows) {
    const columns = Object.keys(rows[0] || {});
    const body = rows.map((row) => columns.map((column) => csvEscape(row[column])).join(",")).join("\n");
    return columns.join(",") + "\n" + body + "\n";
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
    return text;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }

  $("#exportCsvBtn").addEventListener("click", () => {
    download("repaired_sample_review_decisions.csv", toCsv(exportRows()), "text/csv;charset=utf-8");
  });
  $("#exportJsonBtn").addEventListener("click", () => {
    download("repaired_sample_review_decisions.json", JSON.stringify(exportRows(), null, 2), "application/json;charset=utf-8");
  });

  initFilters();
  renderStats();
  applyFilters();
})();
