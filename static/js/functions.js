// send ajax req
function sendReq(callback, url='/load_database') {
	var xhttp = new XMLHttpRequest();
	xhttp.onreadystatechange = function() {
		if (this.readyState == 4 && this.status == 200) {
			callback(this)
		}
	};
	xhttp.open("GET", url, true);
	xhttp.send();
}
