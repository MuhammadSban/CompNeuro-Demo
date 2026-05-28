function calculateBMI() {
    const weight = parseFloat(document.getElementById('weight').value);
    const height = parseFloat(document.getElementById('height').value);
    const inches = parseFloat(document.getElementById('heightin').value);
    const outputbmi = document.getElementById('bmicalcd');

    if(!weight || !height || isNaN(inches))
    {
        outputbmi.textContent = 'FILL OUT ALL FIELDS';
        return;
    }

    const bmiOut = ((weight / (height * 12 + inches) ** 2) * 703).toFixed(2);
    outputbmi.textContent = 'Your BMI: ' + bmiOut;
}