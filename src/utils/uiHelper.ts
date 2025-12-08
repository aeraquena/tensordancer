let countdownEl: HTMLDivElement | null = null;

// Displays and starts countdown
export function startCountdown(seconds: number, isRecording: boolean): void {
  let remaining = seconds;
  // Use the existing countdown element from HTML
  countdownEl = document.getElementById("countdown") as HTMLDivElement | null;

  console.log("start countdown, is recording: ", isRecording);
  if (countdownEl) {
    countdownEl.style.color = isRecording ? "#ffffff" : "#777777";
    countdownEl.textContent = remaining.toString();
  }

  const intervalId = window.setInterval(() => {
    remaining -= 1;
    if (countdownEl) {
      countdownEl.textContent = remaining.toString();
    }
    if (remaining <= 0) {
      clearInterval(intervalId);
      setTimeout(() => {
        if (countdownEl) {
          countdownEl.textContent = "";
        }
      }, 1000);
    }
  }, 1000);
}
