mic_element = document.getElementById('mic');
text_element = document.getElementById('transcription');

function listen(){
	mic_element.style.backgroundColor = 'yellow';
	text_element.innerHTML = "Speak Now...";
	sendReq(listen_callback, '/listen')
}

function listen_callback(req){
	mic_element.style.backgroundColor = '#FFF';

	var result = JSON.parse(req.responseText);
	var output = (result['status'])? result['data'] : 'Could not Resolve query!';
	
	state = parseInt(output);
	if(state == 1){
		location.href = 'http://localhost:3000';
	
	}else{
		text_element.innerHTML = "Try again!";

	}
	mic_element.style.removeProperty('background-color');
}