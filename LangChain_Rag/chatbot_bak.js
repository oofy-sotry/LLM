const chatMessages = document.querySelector('#chat-messages');
const userInput = document.querySelector('#user-input input');
const sendButton = document.querySelector('#user-input button');

userInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendButton.click();
    }
});

sendButton.addEventListener('click', async () => {
    const message = userInput.value.trim();
    if (message.length === 0) return;
    addMessage('나', message);

    userInput.value = '';

    console.log(message);

    try {
        // 1단계: 사용자 입력을 Flask 서버로 전송
        const response = await fetch('http://localhost:5000/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: message })
        });
        console.log("flask서버로 전송 완료");

        if (response.ok) {
            console.log("response.ok");
            // 2단계: Flask 서버에서 받은 검색 결과
            const data = await response.json();
            console.log("flask 서버에서 검색 결과 전송 받음")
            console.log("-----------------------------------문제 데이터가 안넘어옴------------------------------------------")
            console.log(data.query);
            console.log(data.contents);
            if (data.query && data.contents) {
                // 3단계: 검색 결과를 LLM 모델에 전달하여 답변 생성
                const llmResponse = await fetch('http://localhost:5000/generate_answer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        query: data.query,
                        contents: data.contents
                    })
                });
                console.log("LLM 모델로 검색 결과 전달하여 답변 생성후 전달받음")

                if (llmResponse.ok) {
                    const llmData = await llmResponse.json();
                    addMessage('챗봇', llmData.response);
                } else {
                    addMessage('챗봇', 'LLM 응답 오류');
                }
            } else {
                addMessage('챗봇', 'API 호출 중 오류 발생');
            }
        } else {
            const errorData = await response.json();
            //console.error('Error response:', errorData);
            addMessage('챗봇', '서버 오류 발생: ' + errorData.error);
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('챗봇', 'API 호출 중 오류 발생');
    }
});

function addMessage(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender === '나' ? 'user' : 'bot');
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
