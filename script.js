async function translateText() {

    const text = document.getElementById("inputText").value;
    const source = document.getElementById("sourceLang").value;
    const target = document.getElementById("targetLang").value;

    const response = await fetch("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, source, target })
    });

    const data = await response.json();

    document.getElementById("outputText").value = data.translation;
}
