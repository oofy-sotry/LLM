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

    try {
        console.log("Query: " + message);
        console.log("검색 서버로 전달");
        const searchResults = await sendToServer('http://localhost:5000/process', { query: message });
        console.log("Search results:", searchResults);

        if (searchResults && searchResults.contents) {
            const { query, contents } = searchResults;
            console.log("LLM 서버로 전달할 query : ", query);
            const pageContents = contents.map(item => item.page_content);
            console.log("LLM 서버로 전달할 contents : ", pageContents);

            console.log("LLM 서버로 전달")
            const llmResponse = await sendToServer('http://localhost:5000/generate_answer', { query, pageContents });
            console.log("LLM 생성 결과 : ", llmResponse);

            if (llmResponse && llmResponse.response) {
                addMessage('챗봇', llmResponse.response);
            } else {
                addMessage('챗봇', 'LLM 응답이 비어 있습니다.');
            }
        } else {
            addMessage('챗봇', '검색 결과가 없습니다.');
        }
    } catch (error) {
        console.error('Error during API call:', error);
        addMessage('챗봇', '서버와의 통신 중 오류가 발생했습니다.');
    }
});

async function sendToServer(url, body) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        const result = await response.json();
        if (!response.ok) {
            console.error(`Error from server (${url}):`, result);
            return null;
        }
        return result;
    } catch (error) {
        console.error(`Network error for ${url}:`, error);
        return null;
    }
}

function addMessage(sender, message) {
    
    console.log("addMessage의 sender : " + sender);
    console.log("addMessage의 message : " + message);

    const messageElement = document.createElement('div');
    
    const displayMessage = sender === '나' ? '사용자 : ' + message : message;
    
    messageElement.classList.add(sender === '나' ? 'user' : 'bot');
    messageElement.textContent = displayMessage;

    const firstChild = chatMessages.firstChild;
    if (firstChild) {
        chatMessages.insertBefore(messageElement, firstChild);
    } else {
        chatMessages.appendChild(messageElement);
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
}
