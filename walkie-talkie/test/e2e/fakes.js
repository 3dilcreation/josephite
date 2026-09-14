// Browser-side doubles for the two speech APIs, injected before any page script
// runs. Headless Chromium ships neither a recogniser nor an audio output, so
// without these the degraded path cannot be exercised at all.
//
// These stand in for the engines, not for the app: everything between the
// microphone and the far end's speaker is the real implementation.

export function installSpeechFakes({ transcript }) {
  const script = (text) => {
    // --- recognition ---
    class FakeSpeechRecognition {
      constructor() {
        this.lang = 'en-US';
        this.interimResults = false;
        this.continuous = false;
      }

      start() {
        this.running = true;
        // A real engine emits interim results while speech is in progress and
        // finalises shortly after; mimicking that ordering is what makes the
        // test meaningful.
        this._interim = setTimeout(() => {
          if (this.running) this.#emit(text.slice(0, Math.ceil(text.length / 2)), false);
        }, 60);
        this._final = setTimeout(() => {
          if (this.running) this.#emit(text, true);
        }, 200);
      }

      stop() {
        this.running = false;
        clearTimeout(this._interim);
        // A recogniser that has not finalised yet flushes what it has on stop.
        if (!this._finalised) {
          clearTimeout(this._final);
          this.#emit(text, true);
        }
        setTimeout(() => this.onend && this.onend(), 10);
      }

      abort() { this.stop(); }

      #emit(value, isFinal) {
        if (isFinal) this._finalised = true;
        const alternative = { transcript: value, confidence: 0.9 };
        const result = Object.assign([alternative], { isFinal, length: 1 });
        const results = Object.assign([result], { length: 1 });
        this.onresult && this.onresult({ resultIndex: 0, results });
      }
    }
    const define = (name, value) =>
      Object.defineProperty(window, name, { value, configurable: true, writable: true });

    define('SpeechRecognition', FakeSpeechRecognition);
    define('webkitSpeechRecognition', FakeSpeechRecognition);

    // --- synthesis ---
    window.__spoken = [];
    const voices = [
      { name: 'Test Voice A', lang: 'en-US', default: true },
      { name: 'Test Voice B', lang: 'en-GB', default: false },
    ];
    class FakeUtterance {
      constructor(t) { this.text = t; this.voice = null; this.rate = 1; }
    }
    define('SpeechSynthesisUtterance', FakeUtterance);
    // speechSynthesis is a read-only attribute on Window, so a plain assignment
    // silently does nothing and the real (voiceless) engine stays in place.
    define('speechSynthesis', {
      getVoices: () => voices,
      addEventListener() {},
      removeEventListener() {},
      cancel() {},
      speak(utterance) {
        window.__spoken.push({ text: utterance.text, voice: utterance.voice?.name ?? null });
        setTimeout(() => utterance.onend && utterance.onend(), 10);
      },
    });
  };

  return `(${script.toString()})(${JSON.stringify(transcript)})`;
}
