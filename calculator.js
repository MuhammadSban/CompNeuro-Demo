let currentExpression = '';

function pressButton(value) {
    const display = document.getElementById('calcDisp');
    if(value == 'C') {
        currentExpression = '';
        display.textContent = '0';
        return;
    }

    if(value == '=') {
        sendToPython();
        return;
    }

    currentExpression += value;
    display.textContent = currentExpression;
}

async function sendToPython() {
    const display = document.getElementById('calcDisp');

    let operator = null;
    let operatorSymbol = null;

    if(currentExpression.includes('÷')) {
        operator = 'd';
        operatorSymbol = '÷';
        
    } else if(currentExpression.includes('x')) {
        operator = 'm';
        operatorSymbol = 'x';
    } else if(currentExpression.includes('-')) {
        operator = 's';
        operatorSymbol = '-';
    } else if(currentExpression.includes('+')) {
        operator = 'a';
        operatorSymbol = '+';
    }
    
    if (!operator) {
        display.textContent = currentExpression;
        return;
    }


    const parts = currentExpression.split(operatorSymbol);
    const a = parts[0];
    const b = parts[1];
    try
    {
        const response = await fetch ('http://localhost:8000/calc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ first: a, second: b, operation: operator})
        });

        const data = await response.json();

        if (!response.ok) {
            // Python returned an error (e.g. divide by zero)
            display.textContent = data.detail;
            currentExpression = '';
            return;
        }

        display.textContent = data.response;
        currentExpression = String(data.response);

    } catch (err) {
        display.textContent = 'Server offline';
    }
}