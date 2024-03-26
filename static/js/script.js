document.addEventListener("DOMContentLoaded", function () {
  const analyze = document.getElementById("analyze");
  const showDetailsButton = document.getElementById("showDetailsButton");
  const detailSection = document.getElementById("detailSection");
  const processBreakdown = document.getElementById("processBreakdown");

  analyze.addEventListener("click", function () {
    detailSection.style.display = "block";
  });

  showDetailsButton.addEventListener("click", function () {
    processBreakdown.classList.toggle("show");

    // Scroll to the collapsed content
    if (processBreakdown.classList.contains("show")) {
      processBreakdown.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  });
});
