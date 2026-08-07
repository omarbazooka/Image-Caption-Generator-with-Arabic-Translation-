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
const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const BROWSER_CAPTION_MODEL = "Xenova/vit-gpt2-image-captioning";
const BROWSER_TRANSLATION_MODEL = "Xenova/opus-mt-en-ar";

let selectedFile = null;
let previewUrl = null;
let transformersPromise = null;
let browserCaptionerPromise = null;
let browserTranslatorPromise = null;

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

function setLoading(isLoading, message = "") {
  generateButton.disabled = isLoading || !selectedFile;
  buttonText.textContent = isLoading ? "Generating caption..." : "Generate caption";
  spinner.classList.toggle("hidden", !isLoading);

  if (isLoading) {
    setResultState("Running AI models", "loading");
    setStatus(message || "Analyzing the image and generating the bilingual result...");
  }
}

function showResults(english, arabic, sourceLabel) {
  englishResult.textContent = english;
  arabicResult.textContent = arabic;
  englishResult.classList.remove("placeholder");
  arabicResult.classList.remove("placeholder");
  copyButtons.forEach((button) => {
    button.disabled = false;
  });

  setResultState("Caption ready", "ready");
  setStatus(`Caption generated successfully${sourceLabel ? ` · ${sourceLabel}` : ""}.`);
}

async function getServerInferenceMode() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) {
      return "server";
    }
    const payload = await response.json();
    return payload.inference_mode || "server";
  } catch {
    return "browser";
  }
}

async function getTransformers() {
  if (!transformersPromise) {
    transformersPromise = import(TRANSFORMERS_CDN);
  }
  return transformersPromise;
}

async function getBrowserCaptioner() {
  if (!browserCaptionerPromise) {
    browserCaptionerPromise = (async () => {
      const { pipeline } = await getTransformers();
      setStatus("Free hosting mode: downloading the quantized image-caption model to your browser. The first run can take a while...");
      return pipeline("image-to-text", BROWSER_CAPTION_MODEL, {
        dtype: "q8",
        device: "wasm",
      });
    })();
  }
  return browserCaptionerPromise;
}

async function getBrowserTranslator() {
  if (!browserTranslatorPromise) {
    browserTranslatorPromise = (async () => {
      const { pipeline } = await getTransformers();
      setStatus("Image caption ready. Loading the Arabic translation model in your browser...");
      return pipeline("translation", BROWSER_TRANSLATION_MODEL, {
        dtype: "q8",
        device: "wasm",
      });
    })();
  }
  return browserTranslatorPromise;
}

async function runBrowserInference() {
  if (!selectedFile) {
    throw new Error("The selected image is unavailable. Please choose the image again.");
  }

  const { RawImage } = await getTransformers();
  const browserImage = await RawImage.fromBlob(selectedFile);
  const captioner = await getBrowserCaptioner();
  setStatus("Running image captioning locally in your browser...");
  const captionOutput = await captioner(browserImage, { max_new_tokens: 40 });
  const englishCaption = captionOutput?.[0]?.generated_text?.trim();

  if (!englishCaption) {
    throw new Error("The browser caption model did not return a caption.");
  }

  const translator = await getBrowserTranslator();
  setStatus("Translating the generated caption to Arabic locally in your browser...");
  const translationOutput = await translator(englishCaption, { max_new_tokens: 80 });
  const arabicCaption = (
    translationOutput?.[0]?.translation_text ||
    translationOutput?.[0]?.generated_text ||
    ""
  ).trim();

  if (!arabicCaption) {
    throw new Error("The browser translation model did not return an Arabic translation.");
  }

  return {
    english_caption: englishCaption,
    arabic_caption: arabicCaption,
  };
}

async function runServerInference() {
  const formData = new FormData();
  formData.append("image", selectedFile);

  const response = await fetch("/api/caption", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    const error = new Error(payload.detail || "The image could not be processed.");
    error.status = response.status;
    throw error;
  }

  return payload;
}

async function generateCaption() {
  if (!selectedFile) {
    return;
  }

  setLoading(true, "Checking the available inference engine...");

  try {
    const inferenceMode = await getServerInferenceMode();

    if (inferenceMode === "browser") {
      const payload = await runBrowserInference();
      showResults(payload.english_caption, payload.arabic_caption, "browser AI fallback");
      return;
    }

    try {
      setStatus("Analyzing the image with the original BLIP + MarianMT backend...");
      const payload = await runServerInference();
      showResults(payload.english_caption, payload.arabic_caption, "BLIP + MarianMT backend");
    } catch (serverError) {
      if (serverError.status && serverError.status < 500) {
        throw serverError;
      }

      setStatus("The server AI is unavailable on this low-memory instance. Switching to local browser inference...");
      const payload = await runBrowserInference();
      showResults(payload.english_caption, payload.arabic_caption, "browser AI fallback");
    }
  } catch (error) {
    setResultState("Generation failed", "error");
    setStatus(
      error.message ||
        "Browser inference could not start. Check your connection and try again in a modern desktop browser.",
      "error",
    );
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
