const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const emptyUpload = document.getElementById("emptyUpload");
const previewWrap = document.getElementById("previewWrap");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const changeImage = document.getElementById("changeImage");
const generateButton = document.getElementById("generateButton");
const buttonText = document.getElementById("buttonText");
const spinner = document.getElementById("spinner");
const statusMessage = document.getElementById("statusMessage");
const resultState = document.getElementById("resultState");
const englishResult = document.getElementById("englishResult");
const arabicResult = document.getElementById("arabicResult");
const copyButtons = document.querySelectorAll(".copy-button");

const MAX_FILE_SIZE = 8 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

let selectedFile = null;
let previewUrl = null;

function setStatus(message = "", type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status-message${type ? ` ${type}` : ""}`;
}

function setResultState(label, stateClass = "") {
  resultState.textContent = label;
  resultState.className = `state-pill${stateClass ? ` ${stateClass}` : ""}`;
}

function clearResults() {
  englishResult.textContent = "Your generated English caption will appear here.";
  englishResult.classList.add("placeholder");
  arabicResult.textContent = "ستظهر الترجمة العربية هنا.";
  arabicResult.classList.add("placeholder");
  copyButtons.forEach((button) => {
    button.disabled = true;
    button.textContent = "Copy";
  });
  setResultState("Waiting for image");
}

function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return "Please choose a JPG, PNG, or WebP image.";
  }
  if (file.size > MAX_FILE_SIZE) {
    return "Please choose an image smaller than 8 MB.";
  }
  return null;
}

function showFile(file) {
  const validationError = validateFile(file);
  if (validationError) {
    setStatus(validationError, "error");
    return;
  }

  selectedFile = file;
  setStatus("");
  clearResults();

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  fileName.textContent = file.name;
  emptyUpload.classList.add("hidden");
  previewWrap.classList.remove("hidden");
  generateButton.disabled = false;
}

function resetSelection() {
  selectedFile = null;
  fileInput.value = "";
  generateButton.disabled = true;
  emptyUpload.classList.remove("hidden");
  previewWrap.classList.add("hidden");
  setStatus("");
  clearResults();

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function setLoading(isLoading) {
  generateButton.disabled = isLoading || !selectedFile;
  buttonText.textContent = isLoading ? "Generating caption..." : "Generate caption";
  spinner.classList.toggle("hidden", !isLoading);

  if (isLoading) {
    setResultState("Running AI models", "loading");
    setStatus("Analyzing the image with BLIP, then translating the caption with MarianMT...");
  }
}

async function generateCaption() {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

  setLoading(true);

  try {
    const response = await fetch("/api/caption", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(payload.detail || "The image could not be processed.");
    }

    englishResult.textContent = payload.english_caption;
    arabicResult.textContent = payload.arabic_caption;
    englishResult.classList.remove("placeholder");
    arabicResult.classList.remove("placeholder");
    copyButtons.forEach((button) => {
      button.disabled = false;
    });

    setResultState("Caption ready", "ready");
    setStatus("Caption and Arabic translation generated successfully.");
  } catch (error) {
    setResultState("Generation failed", "error");
    setStatus(error.message || "Something went wrong. Please try again.", "error");
  } finally {
    setLoading(false);
  }
}

dropZone.addEventListener("click", (event) => {
  if (event.target.closest("button")) {
    return;
  }
  fileInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) {
    showFile(file);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    showFile(file);
  }
});

changeImage.addEventListener("click", (event) => {
  event.stopPropagation();
  resetSelection();
  fileInput.click();
});

generateButton.addEventListener("click", generateCaption);

copyButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copy);
    if (!target || target.classList.contains("placeholder")) {
      return;
    }

    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = "Copy";
      }, 1300);
    } catch {
      button.textContent = "Select text";
    }
  });
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
});
