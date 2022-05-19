const button = document.querySelector("#speech");
const button1 = document.querySelector("#stream");
let para = document.querySelector("#paras");
let container = document.querySelector('#content');
container.appendChild(para);

window.SpeechRecognition = webkitSpeechRecognition || window.SpeechRecognition;
const synth = window.speechSynthesis;
recognition = new SpeechRecognition();
recognition.interinResult = true;


button.addEventListener('click', () => {
    dictate();
});
const dictate = () => {
    recognition.start();
    recognition.onresult = (event) => {
        const speechToText = Array.from(event.results).map(result => result[0]).map(result => result.transcript).join('');
        console.log(speechToText);
        $.ajax({
            url: '/SpeechRecognition',
            type: 'POST',
            data: { 'message': speechToText },
            success: function(response) {
                const utterThis = new SpeechSynthesisUtterance(response)
                synth.speak(utterThis);

            },
            error: function(error) {
                console.log(error);
            }
        });
        e.preventDefault();


    }

};
button1.addEventListener('click', () => {

    let video = document.querySelector("#video");


    // if there are media devices avaiable get the video on the system
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {

        navigator.mediaDevices.getUserMedia({
            video: true
        }).then(stream => {
            var mediaSource = new MediaSource();
            video.src = URL.createObjectURL(mediaSource);
            video.srcObject = stream;
            video.play();
            $.ajax({
                url: '/VideoStream',
                type: 'POST',
                data: { 'Url': video.src },
                success: function(response) {
                    const utterThis = new SpeechSynthesisUtterance(response)
                    synth.speak(utterThis);

                },
                error: function(error) {
                    console.log(error);
                }
            });
            e.preventDefault();

        })
    }

});




/*const cam = document.querySelector('#myCamera')
Webcam.set({
        width: 320,
        height: 240,
        image_format: 'jpeg',
        jpeg_quality: 90
    }),
    Webcam.attach('')


const dictate = () => {
    recognition.start();
    recognition.onresult = (event) => {
        const speechToText = Array.from(event.results).map(result => result[0]).map(result => result.transcript).join('');
        console.log(speechToText);
        para.textContent = speechToText;
        console.log(event);
        if (speechToText === 'hello world') {
            speak();
        }

        if (event.results[0].isFinal) {
            para = document.createElement('p');
            container.appendChild(para);
        }



    }

};
const speak = () => {
        const utterThis = new SpeechSynthesisUtterance('welcome')
        synth.speak(utterThis);

    } *
    /
    /*
    $('#btnSubmit').click(() => {


    alert("Button worked");
    var email = $('#exampleInputEmail1').val();
    var password = $('#exampleInputPassword1').val();
    alert(`Email: ${email} Password: ${ password }`);

    $.ajax({
        url: '/SpeechRecognition',
        type: 'POST',
        data: { 'Email': email, 'Password': password },
        success: function(r) {
            console.log(r);
        },
        error: function(error) {
            console.log(error);
        }
        e.preventDefault();
    });

    });
    */