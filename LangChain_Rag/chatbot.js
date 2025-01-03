const chatMessages = document.querySelector('#chat-messages');
const userInput = document.querySelector('#user-input input');
const sendButton = document.querySelector('#user-input button');

// Enter 키 입력 시 메시지 전송
userInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendButton.click();
    }
});

// 버튼 클릭 시 메시지 전송
sendButton.addEventListener('click', async () => {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage('나', message);
    userInput.value = '';
    console.log("message : " + message);

    try {
        // Flask 서버로 쿼리 전송
        console.log("flask 서버로 쿼리 전송 시작");
        const searchResults = await sendToServer('http://localhost:5000/process', { query: message });
        console.log("flask 서버로 쿼리 전송 완료");

        if (searchResults) {
            const { query, contents } = searchResults;
            console.log("검색 결과");
            console.log(searchResults);

            // 검색 결과가 있다면 LLM 서버로 전달
            const llmResponse = await sendToServer('http://localhost:5000/generate_answer', { query, contents });
            console.log("llm 서버 전달 완료");

            if (llmResponse) {
                addMessage('챗봇', llmResponse.response || 'LLM 응답이 비어있습니다.');
            } else {
                addMessage('챗봇', 'LLM 응답 처리 중 오류가 발생했습니다.');
            }
        } else {
            addMessage('챗봇', '검색 결과가 없습니다.');
        }
    } catch (error) {
        console.error('Error during API call:', error);
        addMessage('챗봇', '서버와의 통신 중 오류가 발생했습니다.');
    }
});

/**
 * 서버로 데이터를 전송하고 응답을 반환받는 함수
 * @param {string} url - 요청할 서버 URL
 * @param {object} body - 요청에 포함할 데이터
 * @returns {object|null} - 서버 응답 데이터 또는 null
 */
async function sendToServer(url, body) {
    try {
        // 전송하려는 body 객체를 로그로 찍어서 확인
        console.log("Sending data to server:", body);  // body 객체 전체 출력

        // body 객체가 예상대로만 구성되어 있는지 확인
        if (body instanceof Document || body instanceof HTMLElement) {
            console.error("Document or HTMLElement found in body.");
            return null;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error(`Error from ${url}:`, errorData);
            addMessage('챗봇', `서버 오류 발생: ${errorData.error || '알 수 없는 오류'}`);
            return null;
        }

        return await response.json();  // 응답 받기
    } catch (error) {
        console.error(`Network or processing error for ${url}:`, error);
        addMessage('챗봇', '네트워크 또는 처리 오류가 발생했습니다.');
        return null;
    }
}


/**
 * 채팅 메시지를 화면에 추가하는 함수
 * @param {string} sender - 메시지 보낸 사람 ('나' 또는 '챗봇')
 * @param {string} message - 메시지 내용
 */
function addMessage(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender === '나' ? 'user' : 'bot');
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
