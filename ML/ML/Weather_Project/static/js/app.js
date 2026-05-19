document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const predictButton = document.getElementById("predictButton");
    const buttonText = predictButton.querySelector(".button-text");
    const buttonLoader = predictButton.querySelector(".button-loader");
    const errorBanner = document.getElementById("errorBanner");

    const conditionMap = {
        Sunny: {
            bodyClass: "condition-sunny",
            iconClass: "fa-solid fa-sun",
            badge: "Sunny Outlook",
        },
        Rainy: {
            bodyClass: "condition-rainy",
            iconClass: "fa-solid fa-cloud-rain",
            badge: "Rain Watch",
        },
        Cloudy: {
            bodyClass: "condition-cloudy",
            iconClass: "fa-solid fa-cloud",
            badge: "Cloud Cover",
        },
        Storm: {
            bodyClass: "condition-storm",
            iconClass: "fa-solid fa-cloud-bolt",
            badge: "Storm Alert",
        },
        Snow: {
            bodyClass: "condition-snow",
            iconClass: "fa-solid fa-snowflake",
            badge: "Snow Pattern",
        },
        Fog: {
            bodyClass: "condition-fog",
            iconClass: "fa-solid fa-smog",
            badge: "Fog Layer",
        },
    };

    const applyConditionTheme = (condition) => {
        const theme = conditionMap[condition] || conditionMap.Sunny;
        document.body.className = theme.bodyClass;
        document.getElementById("resultIcon").className = `hero-icon ${theme.iconClass}`;
        document.getElementById("conditionTitle").textContent = condition;
        document.getElementById("conditionBadge").textContent = theme.badge;
    };

    const setLoading = (isLoading) => {
        predictButton.disabled = isLoading;
        buttonText.classList.toggle("d-none", isLoading);
        buttonLoader.classList.toggle("d-none", !isLoading);
    };

    const showError = (message) => {
        errorBanner.textContent = message;
        errorBanner.classList.remove("d-none");
    };

    const clearError = () => {
        errorBanner.classList.add("d-none");
        errorBanner.textContent = "";
    };

    const updateMetric = (elementId, value, suffix = "") => {
        document.getElementById(elementId).textContent = `${value}${suffix}`;
    };

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        clearError();
        setLoading(true);

        const payload = Object.fromEntries(new FormData(form).entries());

        try {
            const response = await fetch(form.dataset.apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction request failed.");
            }

            applyConditionTheme(data.climate_condition);
            document.getElementById("summaryText").textContent = data.summary;
            updateMetric("temperatureValue", data.temperature, " deg C");
            updateMetric("humidityValue", data.humidity, "%");
            updateMetric("windValue", data.wind_speed, " km/h");
            updateMetric("pressureValue", data.pressure, " hPa");
            updateMetric("rainProbabilityValue", data.rain_probability, "%");
            updateMetric("confidenceValue", data.confidence_score, "%");
        } catch (error) {
            showError(error.message);
        } finally {
            setLoading(false);
        }
    });

    applyConditionTheme("Sunny");
});
